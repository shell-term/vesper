"""Actual bounded GPQA native attempts with original, dynamically resolved assets.

This opt-in backend does not require a saved answer/route trajectory. Capacity,
missing assets, budgets and errors remain incomplete attempts, not quality passes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_runtime import open_backbone, validate_moe_manifests
from .benchmark_assets import BenchmarkReader, MissingBenchmarkAsset
from .benchmark_attempt import BackendScopeExhausted, run_subset
from .compile_cache import CompilationCache
from .connected import ConnectedMoE
from .expert_bank import ExpertBank
from .formats import bf16_values
from .gate import GateProjection, load_gate_fixture
from .generation_assets import GenerationAssets, reserve_disk
from .gpqa_readiness import load_subset
from .indexed_store import IndexedStore
from .memory import MemoryPolicy, RuntimeMemory
from .model_artifacts import digest
from .output_assets import load as load_output
from .resources import GiB, MiB, MemoryBudget
from .router import RouterProbe, load_router_fixture
from .sequence import MAX_EMBEDDINGS, MAX_ROWS
from .session_benchmark import benchmark_config, validate_images
from .shared_attention import CAPACITY
from .source_roots import source_lease
from .tokenizer import PinnedTokenizer
from .trajectory_workspace import bank_path, bank_sources
from .workspace import Workspace


class DynamicMoE:
    def __init__(self, root, bindings, layer, bank, reader, transfers, scratch, progress):
        from .backbone_runtime import moe_files
        self.root, self.bindings, self.layer = root, bindings, layer
        self.files = moe_files(root, bindings, layer)
        self.bank, self.reader, self.transfers, self.scratch = bank, reader, transfers, scratch
        self.progress = progress; self.calls = 0

    def forward(self, values, image_span=False):
        self.reader.check()
        weight, _ = load_gate_fixture(self.files[0]); bias, _ = load_router_fixture(self.files[1])
        with GateProjection(weight) as gate, RouterProbe(bias) as router:
            route = router.inspect(gate.project(bf16_values(values)), image_span)
        ids = sorted(map(int, route['indices']))
        if self.reader.maximum_fetch_bytes == 0 and any(not self.bank.has(self.layer, e) for e in ids):
            raise MissingBenchmarkAsset('Actual route needs an uncached expert in offline mode')
        self.bank.prepare(self.layer, ids, self.reader, self.transfers)
        self.reader.check()
        with tempfile.TemporaryDirectory(prefix='index-', dir=self.scratch) as tmp:
            target = Path(tmp) / 'index'; self.bank.index(self.layer, ids, target)
            factory = lambda directory, budget: IndexedStore(directory, self.root, budget, layer=self.layer)
            with ConnectedMoE(target, *self.files, config=benchmark_config('host-retained'),
                    layer=self.layer, profile='backbone_probe', source_factory=factory) as moe:
                validate_moe_manifests(moe, self.bindings, self.layer)
                result = moe.forward(values, image_span)
        if (not np.array_equal(result['routing']['indices'], route['indices'])
                or not np.array_equal(result['routing']['weights'].view(np.uint32), route['weights'].view(np.uint32))):
            raise RuntimeError('Dynamic original-route resolution differs from executed routing')
        self.calls += 1
        if self.progress: self.progress({'layer': self.layer, 'completed_layer_calls': self.calls})
        return result


def backend_factory(workspace, directory, *, maximum_fetch_bytes=0, progress=None):
    root = workspace.root / 'artifacts'
    observations = []
    @contextmanager
    def open_backend(case, deadline):
        with ExitStack() as stack:
            reserve_disk(root, 64 * MiB)
            run = directory / ('assets-' + case['id'] + '-' + uuid.uuid4().hex[:8])
            run.mkdir(); scratch = run / 'scratch'; scratch.mkdir()
            observed = {'case_id': case['id'], 'phase': 'setup', 'model_steps_attempted': 0,
                        'completed_model_calls': 0}
            observations.append(observed)
            def write_observation():
                with (run / 'backend.json').open('xb') as stream:
                    stream.write((json.dumps(observed, indent=2) + '\n').encode())
            stack.callback(write_observation)
            allowance = MemoryBudget(cap=12 * GiB, owner='gpqa_native_fixtures')
            allowance.acquire(allowance.cap); stack.callback(allowance.release, allowance.cap)
            cache = stack.enter_context(CompilationCache())
            if progress: progress({'setup': 'compiler_cache_open', 'case_id': case['id']})
            reader = stack.enter_context(BenchmarkReader(root, deadline, maximum_fetch_bytes=maximum_fetch_bytes))
            bank = stack.enter_context(ExpertBank(root, bank_path(root), bank_sources(root)))
            if progress: progress({'setup': 'source_inventory_ready', 'experts': len(bank.entries),
                                   'cached_input_ranges': len(reader.cached)})
            transfers = stack.enter_context(ThreadPoolExecutor(max_workers=8))
            assets = GenerationAssets(root, run, reader, max_embeddings=MAX_EMBEDDINGS, max_rows=MAX_ROWS)
            tokenizer = PinnedTokenizer(root / 'import-source')
            assets.token(tokenizer.encode(case['encoded_prompt'])[0])
            bindings = json.loads((root / 'backbone-controls.json').read_bytes())
            arrays = load_bindings(root, bindings); stack.callback(arrays.clear)
            head_arrays, _ = load_output(root / 'output-fixture'); stack.callback(head_arrays.clear)
            providers = {l: DynamicMoE(root, bindings, l, bank, reader, transfers, scratch, progress) for l in LAYERS}
            engine = open_backbone(stack, root, arrays, assets.token_rows, providers, head_arrays,
                                   temperature=1.0, engram_rows=assets.rows)
            if progress: progress({'setup': 'native_engine_ready'})
            observed['phase'] = 'engine_ready'
            def step(ids, *, sample):
                reader.check()
                if engine.position >= CAPACITY: raise BackendScopeExhausted('Native attention position capacity')
                token = int(ids[0, 0])
                if token not in assets.token_rows and len(assets.token_rows) == MAX_EMBEDDINGS:
                    raise BackendScopeExhausted('Native embedding inventory capacity')
                candidate = engine.transaction.history.prepare(ids, engine.position)
                for i, layer in enumerate((1, 14)):
                    if len(set(assets.rows[layer].entries) | set(map(int, candidate.hash_ids[:, :, i, :].reshape(-1)))) > MAX_ROWS:
                        raise BackendScopeExhausted('Native selected Engram row capacity')
                assets.token(token); assets.prepare_rows(candidate)
                observed['model_steps_attempted'] += 1
                result = engine.step(ids, sample=sample)
                if progress: progress({'model_position': engine.position, 'sample': sample,
                                       'full_vocab_logit_sha256': digest(result['head']['logits_fp32'].tobytes())})
                return result
            backend = SimpleNamespace(tokenizer=tokenizer, step=step, maximum_model_calls=CAPACITY,
                                      allow_partial_capacity=True)
            try:
                yield backend
            finally:
                observed.update({'reader': reader.stats(), 'bank': bank.stats(),
                            'completed_model_calls': engine.position,
                            'layer_calls': {str(l): p.calls for l, p in providers.items()},
                            'compiled_images': cache.manifest(), 'compiler': cache.stats(),
                            'extra_quantization': False, 'routing_or_sampling_change': False,
                            'temperature': 1.0, 'top_p': 1.0, 'seed': 2026,
                            'native_dynamic_asset_import_included_in_timing': True})
                if all(p.calls for p in providers.values()): validate_images(cache.manifest(), 'host-retained')
                observed['phase'] = 'backend_closed'
    open_backend.observations = observations
    return open_backend


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset', default='gpqa-diamond-dev-v1'); p.add_argument('--run', required=True)
    p.add_argument('--budget-seconds', type=float, default=180)
    p.add_argument('--maximum-fetch-bytes', type=int, default=0)
    args = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.run): p.error('Expected a bounded run name')
    w = Workspace.open(Path.cwd()); w.check_runtime_module(__file__, 'src/vesper/gpqa_native.py')
    if w.platform != 'pc-heterogeneous': p.error('Native GPQA backend is PC-only')
    if any(importlib.util.find_spec(n) is not None or n in sys.modules for n in ('torch','tilelang','transformers','sympy')):
        p.error('Use the independent framework-free native environment')
    plan = load_subset(w, args.subset)
    directory = w.artifact_path('benchmarks/' + args.run + '/placeholder', create_parent=True).parent
    context = w.context('pc-' + args.run)
    candidate = {'workspace_context': context, 'numerical_profile': 'released-fp4-dynamic-source-v1',
                 'runtime_settings': {'temperature': 1.0, 'top_p': 1.0, 'seed': 2026,
                    'attention_position_capacity': CAPACITY, 'maximum_embedding_rows': MAX_EMBEDDINGS,
                    'maximum_Engram_rows_per_layer': MAX_ROWS, 'partial_capacity_attempts': True,
                    'maximum_successful_payload_fetch_bytes_per_question': args.maximum_fetch_bytes},
                 'resource_policy': {'host_cap_bytes': 96 * GiB, 'available_floor_bytes': 32 * GiB,
                                     'pinned_cap_bytes': 2 * GiB}}
    progress = lambda row: print(json.dumps(row), flush=True)
    with source_lease(w.root / 'artifacts'):
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=96 * GiB, pinned_cap_bytes=2 * GiB)) as policy:
            factory = backend_factory(w, directory, maximum_fetch_bytes=args.maximum_fetch_bytes, progress=progress)
            report = run_subset(plan, candidate, directory,
                factory,
                budget_seconds=args.budget_seconds)
            report['memory_before_close'] = policy.snapshot()
        report['memory_after_close'] = policy.snapshot()
    report['numerical_profile'] = candidate['numerical_profile']
    report['generation_allowance_unchanged'] = True
    report['backend_observations'] = factory.observations
    report['model_execution_attempted'] = any(o['model_steps_attempted'] for o in factory.observations)
    report['model_execution_run'] = any(r['timing'] and r['timing']['completed_model_calls'] for r in report['results'])
    name = 'attempt-summary-' + uuid.uuid4().hex[:12] + '.json'
    with (directory / name).open('xb') as stream:
        stream.write((json.dumps(w.stamp_result(report, initial_context=context), indent=2) + '\n').encode())
    print(json.dumps({'summary': name, 'completed_questions': len(report['completed_case_ids']),
                      'pending_questions': len(report['pending_case_ids']), 'model_execution_run': bool(report['model_execution_run'])}))


if __name__ == '__main__': main()
