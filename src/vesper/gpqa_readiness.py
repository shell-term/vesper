"""CPU-only readiness recheck of an immutable GPQA subset; no model execution."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .benchmark_attempt import _read
from .checkpoint import MODEL, REVISION
from .gpqa_subset import (ARCHIVE_SHA256, CSV_SHA256, SOURCE_REVISION, POPULATION, PROTOCOL, load_archive,
                          make_cases, preflight)
from .model_artifacts import canonical
from .output_assets import EMBED_IDS
from .sequence_assets import prompt_hashes
from .tokenizer import PinnedTokenizer, compressed_token_map
from .workspace import Workspace


def load_subset(workspace, name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', name):
        raise ValueError('Invalid subset run name')
    plan = _read(workspace.artifact_path('benchmarks/' + name + '/plan.json'))
    body = {k: v for k, v in plan.items() if k not in ('workspace_context', 'plan_sha256')}
    context = plan.get('workspace_context', {})
    if (canonical(body) != plan.get('plan_sha256') or context.get('platform') != workspace.platform
            or context.get('profile') != workspace.profile or context.get('model') != MODEL
            or context.get('revision') != REVISION or plan.get('source_revision') != SOURCE_REVISION
            or plan.get('archive_sha256') != ARCHIVE_SHA256 or plan.get('csv_sha256') != CSV_SHA256
            or plan.get('benchmark') != 'gpqa-diamond' or plan.get('population') != POPULATION
            or plan.get('protocol') != PROTOCOL or plan.get('temperature') != 1.0 or plan.get('top_p') != 1.0
            or plan.get('full_evaluation_run') is not False or type(plan.get('case_count')) is not int
            or not 0 < plan['case_count'] < POPULATION or len(plan.get('cases', [])) != plan['case_count']):
        raise ValueError('Different prepared subset or source identity')
    cases = plan['cases']
    expected = make_cases(load_archive(workspace.artifact_path('benchmarks/gpqa-source/dataset.zip')),
                          plan['case_count'], seed=plan['selection_seed'],
                          max_new_tokens=cases[0]['max_new_tokens'])
    if cases != expected:
        raise ValueError('Prepared questions/settings differ from the pinned source selection')
    return plan


def input_requirements(cases, tokenizer, mapping):
    """Count exact prompt embedding/Engram requirements without running layers.

    Counts describe logical original payloads, not future expert routes, actual
    imports, physical read amplification or complete generation requirements.
    """
    result = []
    for case in cases:
        tokens = tokenizer.encode(case['encoded_prompt'])
        hashes, _ = prompt_hashes(tokens, mapping)
        embeddings = set(tokens)
        rows = {str(layer): len(np.unique(hashes[:, :, i, :])) for i, layer in enumerate((1, 14))}
        result.append({'id': case['id'], 'prompt_tokens': len(tokens),
                       'distinct_prompt_embedding_rows': len(embeddings),
                       'prompt_embedding_payload_bytes': len(embeddings) * 10240,
                       'rows_already_in_legacy_fixed_embedding_fixture': len(embeddings.intersection(EMBED_IDS)),
                       'distinct_prompt_Engram_rows': rows,
                       'prompt_Engram_payload_bytes': sum(rows.values()) * 264,
                       'expert_route_coverage': 'unknown_without_model_execution',
                       'complete_generation_asset_coverage': 'not_established'})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', default='gpqa-diamond-dev-v1')
    parser.add_argument('--record', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.record):
        parser.error('Expected a simple new readiness record name')
    workspace = Workspace.open(Path.cwd())
    workspace.check_runtime_module(__file__, 'src/vesper/gpqa_readiness.py')
    if workspace.platform != 'pc-heterogeneous':
        parser.error('This readiness audit is scoped to the consumer PC')
    plan = load_subset(workspace, args.subset)
    tokenizer = PinnedTokenizer(workspace.root / 'artifacts/import-source')
    mapping, tokenizer_identity = compressed_token_map(tokenizer)
    report = {'probe': 'gpqa_native_readiness', 'prepared_plan_sha256': plan['plan_sha256'],
              'readiness': preflight(plan['cases'], tokenizer), 'tokenizer': tokenizer_identity,
              'prompt_input_requirements': input_requirements(plan['cases'], tokenizer, mapping),
              'model_execution_run': False, 'quality_scored': False, 'GPU_execution_run': False,
              'payload_bytes_fetched_or_copied': 0, 'full_evaluation_run': False,
              'existing_source_payload_integrity_rechecked': False}
    target = workspace.artifact_path('benchmarks/' + args.subset + '/' + args.record + '.json', create_parent=True)
    with target.open('xb') as stream:
        stream.write((json.dumps(workspace.stamp_result(report, run_id='pc-' + args.record), indent=2) + '\n').encode())
    print(json.dumps(report))


if __name__ == '__main__': main()
