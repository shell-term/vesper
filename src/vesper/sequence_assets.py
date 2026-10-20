"""Prepare only longer original input rows; never count preparation as inference."""

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION
from .generation_assets import GenerationAssets, reserve_disk
from .message_cases import quality_cases
from .messages import encode_messages
from .model_artifacts import canonical
from .ngram import NgramState
from .resources import GiB, MiB, MemoryBudget
from .sequence import SequenceRequest, MAX_EMBEDDINGS, MAX_ROWS, scope
from .tokenizer import PinnedTokenizer, compressed_token_map


def request(case_id):
    matches = [c for c in quality_cases() if c["id"] == case_id]
    if len(matches) != 1: raise ValueError("Expected a declared quality request")
    c = matches[0]
    prompt = encode_messages(c["messages"], thinking_mode=c["thinking_mode"], reasoning_effort=c["reasoning_effort"])
    return c, SequenceRequest(prompt, c["max_new_tokens"])


def prompt_hashes(tokens, mapping):
    state = NgramState(mapping); hashes = []
    for token in tokens:
        candidate = state.prepare(np.array([[token]], np.int64))
        hashes.append(candidate.hash_ids.copy()); state.commit(candidate)
    return np.concatenate(hashes, axis=1), state.inspect()


def prepare_input(root, run, *, case_id="reasoning-integer", progress=None):
    root, run = Path(root), Path(run)
    if not run.resolve().is_relative_to((root / "sequence").resolve()): raise ValueError("Expected a separate bounded-sequence directory")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "input-plan.json").exists(): raise FileExistsError("Refusing to replace a complete sequence input plan")
    reserve_disk(run, GiB)
    case, req = request(case_id)
    with ExitStack() as stack:
        budget = MemoryBudget(cap=128 * MiB, owner="sequence_input_rows"); budget.acquire(budget.cap)
        stack.callback(budget.release, budget.cap)
        tokenizer = PinnedTokenizer(root / "import-source"); tokens = req.encode(tokenizer)
        mapping, map_identity = compressed_token_map(tokenizer)
        hashes, state = prompt_hashes(tokens, mapping)
        reader = stack.enter_context(CatalogReader(root / "import-source"))
        assets = GenerationAssets(root, run, reader, max_embeddings=MAX_EMBEDDINGS, max_rows=MAX_ROWS)
        for token in tokens: assets.token(token)
        if progress: progress({"stage": "input_embeddings_ready", "tokens": len(tokens), "distinct_embeddings": len(assets.token_rows)})
        for i, layer in enumerate((1, 14)):
            ids = np.ascontiguousarray(hashes[:, :, i, :]); assets.rows[layer].prepare(ids)
            if progress: progress({"stage": "original_rows_ready", "layer": layer, "rows": len(assets.rows[layer].entries)})
        manifest = assets.freeze()
        report = {"schema_version": 1, "probe": "bounded_sequence_input_plan", "status": "pass", "model": MODEL, "revision": REVISION,
            "scope": scope(), "case": case, "request": {"prompt": req.prompt, "max_new_tokens": req.max_new_tokens, "eos_id": req.eos_id},
            "prompt_token_ids": list(tokens), "original_hash_ids": hashes.tolist(), "hash_state": {
                "position": state["position"], "history": state["history"].tolist()}, "tokenizer": map_identity,
            "assets": manifest, "reader": reader.stats(), "maximum_parallel_row_imports": 8,
            "model_inference_run": False, "quality_scored": False, "new_expert_weight_payloads_fetched": 0}
        report["plan_sha256"] = canonical(report)
        with (run / "input-plan.json").open("xb") as f: f.write((json.dumps(report, indent=2) + "\n").encode())
        return report
