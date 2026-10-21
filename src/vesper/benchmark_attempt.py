"""Bounded benchmark attempts; completed questions survive process restarts.

This controller does not supply a model backend. Tests with synthetic backends
are lifecycle checks, never benchmark quality evidence. A resumed interrupted
question starts again; no KV/RNG checkpoint restoration is implied.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import time

from .generation import TextGeneration
from .gpqa_subset import POPULATION, score_response
from .model_artifacts import canonical
from .session_benchmark import timing_summary

MAX_JSON = 16 * 1024 * 1024


class BackendScopeExhausted(RuntimeError):
    """The declared backend scope ended before the request reached EOS."""


def _write(path, value):
    raw = (json.dumps(value, allow_nan=False, indent=2) + '\n').encode()
    if len(raw) > MAX_JSON:
        raise ValueError('Benchmark record exceeds its explicit metadata bound')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _read(path):
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_JSON:
        raise ValueError('Invalid benchmark record extent or location')
    with path.open('rb') as stream:
        raw = stream.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError('Benchmark record grew beyond its bound')
    return json.loads(raw)


def run_identity(plan, candidate):
    """Bind resumption to inputs, protocol, source, numerics and resource policy."""
    required = ('workspace_context', 'numerical_profile', 'runtime_settings', 'resource_policy')
    if (not isinstance(candidate, dict) or any(k not in candidate for k in required)
            or not isinstance(candidate['workspace_context'], dict)):
        raise ValueError('Expected an explicit candidate execution identity')
    cases = plan['cases']
    if (type(plan.get('case_count')) is not int or plan['case_count'] != len(cases)
            or not 0 < len(cases) < POPULATION or plan.get('full_evaluation_run') is not False):
        raise ValueError('This controller only accepts a declared development subset')
    identities = []
    for case in cases:
        if not re.fullmatch(r'gpqa-[a-f0-9]{20}', case.get('id', '')):
            raise ValueError('Invalid benchmark case identity')
        body = {k: v for k, v in case.items() if k != 'case_sha256'}
        if canonical(body) != case.get('case_sha256'):
            raise ValueError('Benchmark question/settings changed')
        identities.append({'id': case['id'], 'sha256': case['case_sha256']})
    if len({c['id'] for c in identities}) != len(cases):
        raise ValueError('Duplicate benchmark case')
    candidate = json.loads(json.dumps(candidate, allow_nan=False))
    candidate['workspace_context'].pop('created_at', None)
    return {'schema_version': 1, 'plan_sha256': canonical(plan),
            'cases': identities, 'candidate': candidate,
            'resume_interrupted_question': 'restart_from_original_prompt'}


class AttemptLedger:
    """Exclusive writer and immutable attempts under one private run directory."""
    def __init__(self, directory, identity):
        self.directory, self.identity = Path(directory), identity
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError('Benchmark run cannot be a symlink')
        self.lock = (self.directory / 'writer.lock').open('a+b')
        try:
            self.lock.seek(0, 2)
            if not self.lock.tell(): self.lock.write(b'0'); self.lock.flush()
            self.lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / 'identity.json'
            if path.exists():
                if _read(path) != self.identity:
                    raise ValueError('Cannot resume with changed input, candidate, settings or resources')
            else:
                _write(path, self.identity)
            return self
        except BaseException:
            self.lock.close(); self.lock = None; raise

    def __exit__(self, *_):
        if self.lock is not None:
            self.lock.close(); self.lock = None

    def attempts(self, case_id):
        if self.lock is None or case_id not in {c['id'] for c in self.identity['cases']}:
            raise ValueError('Expected a locked, declared benchmark question')
        paths = sorted(self.directory.glob(case_id + '-*'))
        if len(paths) > 10000 or any(p.is_symlink() or not p.is_dir()
                or not re.fullmatch(re.escape(case_id) + r'-[0-9]{5}', p.name) for p in paths):
            raise ValueError('Invalid or excessive benchmark attempt history')
        return paths

    def completed(self, case_id):
        result = []
        for path in self.attempts(case_id):
            final = path / 'final.json'
            if final.exists():
                row = _read(final)
                if row.get('case_id') != case_id or row.get('identity_sha256') != canonical(self.identity):
                    raise ValueError('Benchmark attempt belongs to a different run')
                # Wrong answers are completed too; never retry until correct.
                if row['stop_reason'] in ('eos', 'max_new_tokens'):
                    result.append(row)
        if len(result) > 1:
            raise ValueError('Duplicate completed question in a single-trial run')
        return result[0] if result else None

    @contextmanager
    def begin(self, case_id):
        if self.completed(case_id) is not None:
            raise ValueError('Refusing to rerun a completed question')
        paths = self.attempts(case_id)
        ordinal = max([int(p.name[-5:]) for p in paths], default=0) + 1
        path = self.directory / f'{case_id}-{ordinal:05d}'
        path.mkdir(exist_ok=False)
        _write(path / 'started.json', {'case_id': case_id, 'identity_sha256': canonical(self.identity),
                                     'attempt': ordinal, 'runtime_state_checkpointed': False})
        # One append per successful model call. A crash preserves the prefix;
        # it never manufactures an EOS or a completed/graded question.
        with (path / 'calls.jsonl').open('x', encoding='utf-8') as journal:
            yield path, journal


class BenchmarkRequest:
    """Preserve the declared budget without extending a backend's capacity."""
    def __init__(self, case):
        self.prompt, self.max_new_tokens, self.eos_id = case['encoded_prompt'], case['max_new_tokens'], 1

    def encode(self, tokenizer):
        if (not isinstance(self.prompt, str) or not self.prompt
                or type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 262144):
            raise ValueError('Invalid complete benchmark prompt or generation allowance')
        ids = tokenizer.encode(self.prompt)
        if not ids or any(type(i) is not int or not 0 <= i < 129280 for i in ids):
            raise ValueError('Invalid encoded benchmark prompt')
        return tuple(ids)


def run_subset(plan, candidate, directory, open_backend, *, budget_seconds, clock=time.perf_counter):
    """Run once per question; persist timeout/error attempts and skip completions.

    open_backend(case, deadline) is a context manager yielding a backend with
    tokenizer, step and maximum_model_calls. It must enforce the same deadline
    within long model calls and retain the declared candidate/resource policy.
    Setup, scoring, journal writes and cleanup count toward elapsed work.
    """
    if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or not 0 < budget_seconds <= 1200:
        raise ValueError('Expected an explicit bounded subset execution budget')
    identity = run_identity(plan, candidate)
    started = clock(); deadline = started + budget_seconds; results = []
    with AttemptLedger(directory, identity) as ledger:
        for case in plan['cases']:
            old = ledger.completed(case['id'])
            if old is not None:
                results.append(old); continue
            if clock() >= deadline: break
            with ledger.begin(case['id']) as (path, journal):
                item_start = clock(); session = None; durations = []
                stop, error, setup, ttft, text = None, None, None, None, None
                first_ready = last_ready = None
                capacity, partial = None, False
                try:
                    with open_backend(case, deadline) as backend:
                        if clock() >= deadline: raise TimeoutError('Subset setup exhausted its cooperative budget')
                        session = TextGeneration(backend.tokenizer, backend.step, BenchmarkRequest(case))
                        calls = len(session.prompt_ids) + session.request.max_new_tokens - 1
                        partial = getattr(backend, 'allow_partial_capacity', False)
                        capacity = backend.maximum_model_calls
                        if (type(backend.maximum_model_calls) is not int or backend.maximum_model_calls < 1
                                or type(partial) is not bool
                                or calls > backend.maximum_model_calls and not partial):
                            raise ValueError('Backend cannot admit the full prompt/generation allowance')
                        setup = clock() - item_start
                        while not session.done:
                            if clock() >= deadline: raise TimeoutError('Subset reached its cooperative time budget')
                            if session.position >= backend.maximum_model_calls:
                                raise BackendScopeExhausted('Native scope exhausted before request completion')
                            call_start = clock(); meta, result = session.advance()
                            elapsed = clock() - call_start
                            if not math.isfinite(elapsed) or elapsed <= 0:
                                raise ValueError('Expected a positive finite model-call duration')
                            durations.append(elapsed)
                            journal.write(json.dumps({**meta, 'forward_seconds': elapsed}) + '\n')
                            journal.flush(); os.fsync(journal.fileno())
                            if meta['sampling_requested']:
                                last_ready = clock()
                                if first_ready is None:
                                    first_ready = last_ready; ttft = first_ready - item_start
                            result = None  # Do not retain diagnostic tensors between calls.
                        stop = session.stop_reason
                        text = backend.tokenizer.decode(list(session.generated))
                except BackendScopeExhausted:
                    stop = 'backend_scope_exhausted'
                except TimeoutError:
                    stop = 'budget_exceeded' if clock() >= deadline else 'backend_timeout'
                except KeyboardInterrupt:
                    stop = 'interrupted'
                except Exception as failure:
                    stop, error = 'error', type(failure).__name__
                decode_wall = None if first_ready is None else last_ready - first_ready
                decode_count = max(0, len(session.generated) - 1) if session else 0
                row = {'case_id': case['id'], 'identity_sha256': canonical(identity), 'stop_reason': stop,
                       'declared_max_new_tokens': case['max_new_tokens'],
                       'backend_maximum_model_calls': capacity, 'partial_capacity_attempt': partial,
                       'error_type': error, 'selected_ids_including_eos': list(session.generated) if session else [],
                       'decoded_selected_ids': text, 'score': score_response(case, text, stop_reason=stop),
                       'timing': timing_summary(durations, len(session.prompt_ids)) if session else None,
                       'setup_seconds': setup, 'engine_TTFT_seconds': ttft,
                       'request_TTFT_seconds': ttft - setup if ttft is not None else None,
                       'decode_wall_seconds_including_journaling': decode_wall,
                       'accepted_decode_tokens_after_first_including_eos': decode_count,
                       'accepted_decode_wall_tokens_per_second': decode_count / decode_wall if decode_wall else None,
                       'elapsed_seconds_including_setup_scoring_cleanup': clock() - item_start,
                       'time_budget_is_OS_deadline': False, 'runtime_state_checkpointed': False}
                _write(path / 'final.json', row); results.append(row)
                if stop not in ('eos', 'max_new_tokens'): break
        completed = [r['case_id'] for r in results if r['stop_reason'] in ('eos', 'max_new_tokens')]
        return {'identity_sha256': canonical(identity), 'results': results, 'completed_case_ids': completed,
                'pending_case_ids': [c['id'] for c in plan['cases'] if c['id'] not in completed],
                'elapsed_seconds': clock() - started, 'full_evaluation_run': False}
