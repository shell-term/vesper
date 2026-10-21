"""Pinned GPQA Diamond development inputs and honest native-engine preflight.

No inference, model API, full-suite execution or quality qualification is
performed here. Dataset questions/answers stay in ignored platform artifacts.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from .messages import EOS, encode_messages, parse_completion
from .model_artifacts import canonical, digest
from .sequence import MAX_CALLS
from .workspace import Workspace

SOURCE_REVISION = '56686c06f5e19865c153de0fdb11be3890014df7'
ARCHIVE_SHA256 = '461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc'
CSV_SHA256 = '41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305'
POPULATION = 198
PROTOCOL = 'vesper-gpqa-zero-shot-thinking-v1'
FIELDS = ('Record ID', 'Question', 'Correct Answer', 'Incorrect Answer 1',
          'Incorrect Answer 2', 'Incorrect Answer 3')


def _rows(raw):
    """Project only evaluation fields; never include explanations in prompts."""
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not set(FIELDS).issubset(reader.fieldnames or ()):
        raise ValueError('GPQA CSV is missing required evaluation fields')
    result, seen = [], set()
    for row in reader:
        item = {k: row[k] for k in FIELDS}
        if any(type(v) is not str or not v.strip() for v in item.values()):
            raise ValueError('Empty GPQA identity/question/answer')
        if item['Record ID'] in seen:
            raise ValueError('Duplicate GPQA record identity')
        seen.add(item['Record ID']); result.append(item)
    return result


def load_archive(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('GPQA archive exceeds its bounded source size')
    raw = path.read_bytes()
    if digest(raw) != ARCHIVE_SHA256:
        raise ValueError('GPQA archive differs from the pinned upstream bytes')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo('dataset/gpqa_diamond.csv')
        if info.file_size > 2 * 1024 * 1024:
            raise ValueError('GPQA evaluation CSV exceeds its bound')
        data = archive.read(info, pwd=b'deserted-untie-orchid')
    if digest(data) != CSV_SHA256:
        raise ValueError('GPQA CSV differs from the pinned upstream bytes')
    rows = _rows(data)
    if len(rows) != POPULATION:
        raise ValueError('Unexpected GPQA Diamond population')
    return rows


def make_cases(rows, count, *, seed, max_new_tokens):
    if type(count) is not int or not 1 <= count < len(rows):
        raise ValueError('This command prepares a proper subset; full evaluation needs a separate approved plan')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Expected an explicit bounded selection seed')
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 262144:
        raise ValueError('Expected an explicit generation budget, without silently clipping it')
    rank = lambda row: hashlib.sha256(f"{seed}:{row['Record ID']}".encode()).digest()
    selected = sorted(rows, key=rank)[:count]
    cases = []
    for row in selected:
        identity = canonical(row)
        order = sorted(range(4), key=lambda i: digest(f'{seed}:{identity}:choice:{i}'.encode()))
        choices = [row[FIELDS[i + 2]] for i in order]
        content = ('Choose the correct answer to the following question. '
                   'Give only its letter in parentheses as your final answer.\n\n'
                   + row['Question'] + '\n\n'
                   + '\n'.join(f'({letter}) {answer}' for letter, answer in zip('ABCD', choices)))
        case = {'id': 'gpqa-' + identity[:20], 'source_record_id': row['Record ID'],
                'source_item_sha256': identity, 'choice_original_indices': order,
                'source_has_duplicate_choice_text': len(set(choices)) != 4,
                'messages': [{'role': 'user', 'content': content}],
                'thinking_mode': 'thinking', 'reasoning_effort': 75,
                'max_new_tokens': max_new_tokens,
                'scoring': {'kind': 'strict_final_choice', 'expected': 'ABCD'[order.index(0)]}}
        try:
            case['encoded_prompt'] = encode_messages(case['messages'], thinking_mode='thinking', reasoning_effort=75)
            case['prompt_encoding_error'] = None
        except ValueError as error:
            case['encoded_prompt'] = None
            case['prompt_encoding_error'] = str(error)
        case['case_sha256'] = canonical(case)
        cases.append(case)
    return cases


def preflight(cases, tokenizer):
    from .shared_attention import CAPACITY
    from .tokenizer import MAX_DECODE_IDS
    result = []
    for case in cases:
        reasons = []
        if case['source_has_duplicate_choice_text']:
            reasons.append('duplicate_choice_text_in_original_source_not_repaired')
        prompt = case['encoded_prompt']; tokens = None
        if prompt is None:
            reasons.append('message_prompt_exceeds_current_adapter_scope')
        else:
            tokens = tokenizer.encode(prompt)
            if len(tokens) + case['max_new_tokens'] - 1 > MAX_CALLS:
                reasons.append('prompt_and_generation_budget_exceed_native_512_call_scope')
            if len(tokens) + case['max_new_tokens'] - 1 > CAPACITY:
                reasons.append('prompt_and_generation_budget_exceed_native_attention_capacity')
        if case['max_new_tokens'] > MAX_DECODE_IDS:
            reasons.append('completion_budget_exceeds_native_tokenizer_decode_capacity')
        reasons.append('native_dynamic_benchmark_asset_provider_not_implemented')
        # Input admission alone cannot establish an executable benchmark path.
        reasons.append('native_benchmark_inference_adapter_not_implemented')
        result.append({'id': case['id'], 'case_sha256': case['case_sha256'],
                       'source_record_id': case['source_record_id'],
                       'prompt_tokens': None if tokens is None else len(tokens),
                       'max_new_tokens': case['max_new_tokens'],
                       'maximum_model_calls': None if tokens is None else len(tokens) + case['max_new_tokens'] - 1,
                       'ready_for_native_inference': False, 'reasons': reasons})
    return {'status': 'not_ready', 'cases': result, 'native_maximum_model_calls': MAX_CALLS,
            'native_attention_position_capacity': CAPACITY, 'native_maximum_decode_ids': MAX_DECODE_IDS,
            'benchmark_attempt_controller_implemented': True,
            'native_dynamic_asset_resolution_implemented': False,
            'model_execution_run': False, 'quality_scored': False,
            'official_score_reproduction': False, 'full_evaluation_run': False}


def score_response(case, text, *, stop_reason):
    """A local final-letter grader, not proof of a model execution's provenance."""
    if stop_reason != 'eos':
        return {'status': 'incomplete', 'passed': None, 'reason': 'generation_incomplete'}
    if type(text) is not str or not text.endswith(EOS):
        return {'status': 'fail', 'passed': False, 'reason': 'missing_actual_eos'}
    if case['source_has_duplicate_choice_text']:
        return {'status': 'incomplete', 'passed': None, 'reason': 'ambiguous_source_choices'}
    # Grade only the final channel. Long thinking must not be rejected by the
    # small message adapter's diagnostic response-length bound.
    if case['thinking_mode'] == 'thinking':
        _, separator, final = text.partition('</think>')
        if not separator:
            return {'status': 'fail', 'passed': False, 'reason': 'missing_thinking_terminator'}
    else:
        final = text
    try:
        parsed = parse_completion(final, thinking_mode='chat')
    except ValueError:
        return {'status': 'fail', 'passed': False, 'reason': 'malformed_completion'}
    match = re.fullmatch(r'(?:([ABCD])|\(([ABCD])\))', parsed['content'].strip())
    choice = None if match is None else match.group(1) or match.group(2)
    passed = not parsed['tool_calls'] and choice == case['scoring']['expected']
    return {'status': 'pass' if passed else 'fail', 'passed': passed, 'choice': choice}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.run):
        parser.error('Expected a simple subset run name')
    w = Workspace.open(Path.cwd())
    w.check_runtime_module(__file__, 'src/vesper/gpqa_subset.py')
    if w.platform != 'pc-heterogeneous':
        parser.error('This preparation is scoped to the consumer PC')
    source = w.artifact_path('benchmarks/gpqa-source/dataset.zip')
    if args.archive.resolve(strict=True) != source.resolve(strict=True):
        parser.error('Expected the pinned archive inside this PC workspace')
    cases = make_cases(load_archive(source), args.count, seed=args.seed, max_new_tokens=args.max_new_tokens)
    from .tokenizer import PinnedTokenizer
    checked = preflight(cases, PinnedTokenizer(w.root / 'artifacts/import-source'))
    plan = {'benchmark': 'gpqa-diamond', 'source_revision': SOURCE_REVISION,
            'archive_sha256': ARCHIVE_SHA256, 'csv_sha256': CSV_SHA256,
            'population': POPULATION, 'selection_seed': args.seed, 'protocol': PROTOCOL,
            'temperature': 1.0, 'top_p': 1.0, 'case_count': len(cases), 'cases': cases,
            'preflight': checked, 'model_execution_run': False, 'quality_scored': False,
            'official_score_reproduction': False, 'full_evaluation_run': False}
    plan['plan_sha256'] = canonical(plan)
    stamped = w.stamp_result(plan, run_id='pc-' + args.run)
    target = w.artifact_path('benchmarks/' + args.run + '/plan.json', create_parent=True)
    with target.open('xb') as stream:
        stream.write((json.dumps(stamped, indent=2) + '\n').encode())
    print(json.dumps({'case_count': len(cases), 'plan_sha256': plan['plan_sha256'], **checked}))


if __name__ == '__main__':
    main()
