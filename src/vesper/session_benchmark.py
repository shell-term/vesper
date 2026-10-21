"""Opt-in, time-bounded PC generation timing against an existing short reference."""

import argparse
from contextlib import ExitStack
import gc
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import numpy as np

from .attention_validation import array_digest
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_runtime import open_backbone
from .backbone_validation import native_state
from .compile_cache import CompilationCache
from .compile_cache_validation import image_identity, QUALIFIED_CUBIN_IMAGE_SET
from .connected import ConnectedConfig
from .engram_rows import EngramRows
from .generation import TextGeneration
from .generation_assets import load_token_rows, reserve_disk
from .generation_validation import REQUEST, load_plan, plan_identity, validate_original, validate_memory
from .indexed_store import build_index
from .memory import MemoryPolicy, RuntimeMemory
from .model_artifacts import digest
from .moe_session import SessionMoE
from .output_assets import load as load_head
from .resources import GiB, MiB, MemoryBudget
from .source_roots import source_scope
from .tiered import CacheConfig
from .tokenizer import PinnedTokenizer
from .topology_validation import state_digest
from .trajectory_evidence import read
from .workspace import Workspace


def timing_summary(durations, prompt_tokens):
    if (type(prompt_tokens) is not int or prompt_tokens < 1
            or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in durations)):
        raise ValueError('Expected positive finite model-call times and a complete prompt length')
    prefill = sum(durations[:prompt_tokens])
    decode = durations[prompt_tokens:]
    return {'completed_model_calls': len(durations), 'prompt_tokens': prompt_tokens,
            'prompt_complete': len(durations) >= prompt_tokens,
            'prefill_forward_seconds': prefill,
            'decode_tokens_after_first': len(decode), 'decode_forward_seconds': sum(decode),
            'decode_tokens_per_second': len(decode) / sum(decode) if decode else None,
            'forward_seconds': sum(durations)}


def compare_step(meta, result, rng_offset, expected):
    logits = result['head']['logits_fp32']
    checks = [b for b in expected['boundaries'] if b['check'] == 'result/head/logits_fp32/']
    if len(checks) != 1:
        raise ValueError('Missing unique original full-vocabulary logits')
    actual = {'check': checks[0]['check'], 'dtype': str(logits.dtype), 'shape': list(logits.shape),
              'elements': int(logits.size), 'sha256': array_digest(logits)}
    routes = {str(l): {'indices': result[f'layer{l}']['moe_gate.indices'].tolist(),
                      'weight_bits': result[f'layer{l}']['moe_gate.weights'].view(np.uint32).tolist()} for l in LAYERS}
    if (meta != expected['meta'] or actual != checks[0] or routes != expected['ordered_routes']
            or rng_offset != expected['rng_offset']):
        raise RuntimeError('Released-precision short-reference logits, routes, feedback or RNG differ')
    return {'meta': meta, 'logits': actual, 'ordered_routes': routes, 'rng_offset': rng_offset,
            'matches_reference': True}


def prepare(root, directory, original, raw):
    story = root / 'generation/story-v1'
    plan = load_plan(root, story)
    if plan_identity(plan) != original['plan_sha256']:
        raise ValueError('Legacy short input differs from the qualified reference')
    reports = {}
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=2 * GiB)) as policy:
        for layer in LAYERS:
            p = plan['experts'][str(layer)]
            reports[str(layer)] = build_index([root / p['store']], directory / f'indices/layer{layer}',
                                              root, layer=layer, experts=p['ids'])
        before = policy.snapshot()
    return {'probe': 'pc_session_benchmark_input_preparation', 'status': 'pass',
            'model_execution_run': False, 'legacy_short_input_reused_explicitly': True,
            'reference_report_sha256': digest(raw), 'plan_sha256': plan_identity(plan),
            'indices': reports, 'expert_payload_bytes_copied': 0, 'weight_payloads_fetched': 0,
            'memory_before_policy_close': before, 'memory_after_policy_close': policy.snapshot()}


def run(root, directory, original, raw, mode, budget_seconds, progress):
    frameworks = {n: {'installed': importlib.util.find_spec(n) is not None, 'imported': n in sys.modules}
                  for n in ('torch', 'tilelang', 'transformers', 'sympy')}
    if any(any(v.values()) for v in frameworks.values()):
        raise ValueError('Expected the framework-free PC native environment')
    story = root / 'generation/story-v1'
    plan = load_plan(root, story)
    prepared, _ = read(directory / 'preparation.json')
    if (plan_identity(plan) != original['plan_sha256'] or prepared['plan_sha256'] != original['plan_sha256']
            or prepared['reference_report_sha256'] != digest(raw)):
        raise ValueError('Different prepared short-reference input')
    # Both variants use identical capacities. Only component lifetime changes.
    config = ConnectedConfig(cache=CacheConfig(host_cap_bytes=256 * MiB, ram_cache_bytes=6 * 18800640,
        primary_slots=2, helper_slots=2, integrity_mode='gpu-sha256', max_store_experts=48,
        profile=True), host_cap_bytes=448 * MiB)
    providers, calls, durations = {}, [], []
    cache = session = engine = http = None
    setup = ttft = request_ttft = None
    timed_out, final_state = False, None
    hardware = []
    started = time.perf_counter()
    deadline = started + budget_seconds
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=64 * GiB)) as policy:
        try:
            with ExitStack() as stack:
                http = stack.enter_context(patch('urllib.request.urlopen', side_effect=RuntimeError('Benchmark must stay offline')))
                fixtures = MemoryBudget(cap=12 * GiB, owner='session_benchmark_fixtures')
                fixtures.acquire(fixtures.cap)
                stack.callback(fixtures.release, fixtures.cap)
                cache = stack.enter_context(CompilationCache())
                bindings = json.loads((root / 'backbone-controls.json').read_bytes())
                arrays = load_bindings(root, bindings)
                head_arrays, head_manifest = load_head(root / 'output-fixture')
                stack.callback(arrays.clear)
                stack.callback(head_arrays.clear)
                tokens = load_token_rows(root, story, plan)
                rows = {l: stack.enter_context(EngramRows(story / f'rows/layer{l}', l)) for l in (1, 14)}
                for layer in LAYERS:
                    p = stack.enter_context(SessionMoE(root, bindings, layer, plan['experts'][str(layer)],
                        original['expert_provenance'], retain=mode == 'retained', config=config,
                        index_directory=directory / f'indices/layer{layer}', deadline=deadline))
                    providers[str(layer)] = p
                    if p.manifest_sha256 != prepared['indices'][str(layer)]['manifest_sha256']:
                        raise ValueError('Benchmark index changed after preparation')
                engine = open_backbone(stack, root, arrays, tokens, {int(l): p for l, p in providers.items()},
                                       head_arrays, engram_rows=rows)
                if (bindings['bindings_sha256'] != original['bindings_sha256'] or head_manifest != original['output_fixture']
                        or engine.tokenizer_identity != original['tokenizer'] or engine.engram_manifests != original['Engram_fixtures']
                        or {str(l): p.converted_sha256 for l, p in engine.probe_operators['projections'].items()}
                        != original['original_grouped_conversion']):
                    raise ValueError('Benchmark control/source/conversion bindings differ')
                devices = engine.hc[0].device.driver.inventory()['devices']
                hardware = [{k: d[k] for k in ('ordinal', 'name', 'compute_capability')}
                            for d in devices if d['ordinal'] in (0, 1)]
                session = TextGeneration(PinnedTokenizer(root / 'import-source'), engine.step, REQUEST)
                request_start = time.perf_counter()
                setup = request_start - started
                while not session.done:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Session benchmark reached its cooperative time budget')
                    step_start = time.perf_counter()
                    meta, result = session.advance()
                    step_end = time.perf_counter()
                    durations.append(step_end - step_start)
                    if meta['sampling_requested'] and ttft is None:
                        ttft, request_ttft = step_end - started, step_end - request_start
                    checked = compare_step(meta, result, engine.transaction.sampler.offset, original['cases'][len(calls)])
                    checked['forward_seconds'] = durations[-1]
                    calls.append(checked)
                    progress({'mode': mode, 'completed_call': len(calls), 'meta': meta,
                              'forward_seconds': durations[-1], 'reference_match': True})
                if session.summary() != original['completion']:
                    raise RuntimeError('Benchmark completion differs from the original')
                final_state = state_digest(native_state(engine))
                if final_state != original['cases'][-1]['state_sha256']:
                    raise RuntimeError('Benchmark final model/hash/RNG state differs')
                images, compiler = cache.manifest(), cache.stats()
                if image_identity(images) != QUALIFIED_CUBIN_IMAGE_SET:
                    raise RuntimeError('Benchmark compiled images differ from the qualified native set')
        except TimeoutError:
            if time.perf_counter() < deadline:
                raise
            timed_out = True
            images, compiler = [], None
        result = None
        session = None
        engine = None
        gc.collect()
        before = policy.snapshot()
    report = {'schema_version': 1, 'probe': 'pc_short_generation_session_benchmark',
        'status': 'budget_exceeded' if timed_out else 'pass', 'mode': mode,
        'numerical_profile': 'released-fp4', 'additional_quantization': False,
        'changed_routing_or_sampling': False, 'native_frameworks': frameworks,
        'scope': {'case_id': 'raw-text-story-five-calls', 'request': original['scope']['request'],
                  'layers': list(LAYERS), 'maximum_model_calls': 5, 'complete_task_quality_evaluation': 'not_run',
                  'arbitrary_prompt_or_production_throughput_claim': False, 'retraining': False},
        'reference_report_sha256': digest(raw), 'original_identity_sha256': original['identity_sha256'],
        'complete_goldens_sha256': original['complete_goldens_sha256'],
        'new_original_reference_execution_run': False, 'legacy_short_input_reused_explicitly': True,
        'hardware': hardware, 'cooperative_budget_seconds': budget_seconds,
        'time_budget_is_OS_deadline': False, 'all_intermediate_fingerprints_recorded': False,
        'diagnostic_intermediate_tensor_materialization_included': True,
        'timing_scope': {'includes_native_engine_setup_in_engine_TTFT': True,
                         'python_import_and_reference_validation_included': False,
                         'per_call_output_checks_excluded_from_forward_durations': True,
                         'request_TTFT_includes_between_call_output_checks': True,
                         'cold_process_and_contexts': True, 'OS_or_driver_caches_reset': False,
                         'within_request_cache_reuse': mode == 'retained', 'repeated_measurement_claim': False},
        'timing': {**timing_summary(durations, len(original['completion']['prompt_ids'])),
                   'setup_seconds': setup, 'engine_TTFT_seconds': ttft, 'request_TTFT_seconds': request_ttft,
                   'total_engine_wall_seconds_including_checks_and_cleanup': time.perf_counter() - started},
        'calls': calls, 'completion': original['completion'] if not timed_out else None,
        'final_state_sha256': final_state, 'complete_short_reference_match': not timed_out,
        'providers': {l: p.stats() for l, p in providers.items()},
        'compiler_cache': compiler, 'compiled_images': images,
        'compiler_cache_after_close': cache.stats() if cache is not None else None,
        'HTTP_calls': http.call_count if http is not None else 0, 'weight_payloads_fetched': 0,
        'memory_before_policy_close': before, 'memory_after_policy_close': policy.snapshot()}
    validate_memory(report, 64 * GiB)
    for p in report['providers'].values():
        if p['opened'] or p['context_opens'] != p['context_closes'] or any((p['last_cleanup'] or {}).values()):
            raise RuntimeError('Session benchmark retained unfinished component resources')
    if report['HTTP_calls']:
        raise RuntimeError('Session benchmark attempted HTTP')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'baseline', 'retained'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--run', default='session-smoke-v1')
    parser.add_argument('--budget-seconds', type=float, default=300)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.run):
        parser.error('Expected a simple benchmark run name')
    if not math.isfinite(args.budget_seconds) or not 1 <= args.budget_seconds <= 900:
        parser.error('Budget must be between one and 900 cooperative seconds')
    workspace = Workspace.open(Path.cwd())
    workspace.check_runtime_module(__file__, 'src/vesper/session_benchmark.py')
    root = args.artifacts.resolve(strict=True)
    if root != workspace.root / 'artifacts':
        parser.error('Expected this bound PC workspace artifact root')
    directory = workspace.artifact_path(f'performance/{args.run}/preparation.json', create_parent=True).parent
    name = 'preparation.json' if args.mode == 'prepare' else args.mode + '.json'
    target = directory / name
    if target.exists():
        raise FileExistsError('Refusing to replace an existing benchmark result')
    context = workspace.context('pc-' + args.run + '-' + args.mode)
    original, raw = read(workspace.root / 'results/2026-10-19-generation/original.json')
    validate_original(original)
    reserve_disk(directory, 4 * MiB)
    with source_scope(root) as storage:
        report = (prepare(root, directory, original, raw) if args.mode == 'prepare' else
                  run(root, directory, original, raw, args.mode, args.budget_seconds,
                      lambda value: print(json.dumps(value), flush=True)))
        report['source_storage'] = storage.public()
    stamped = workspace.stamp_result(report, initial_context=context)
    with target.open('xb') as stream:
        stream.write((json.dumps(stamped, indent=2) + '\n').encode())
    print(json.dumps({'status': report['status'], 'mode': args.mode,
                      'model_execution_run': args.mode != 'prepare', 'result': name}), flush=True)


if __name__ == '__main__':
    main()
