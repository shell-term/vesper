"""Prepare a bounded generated trajectory; never count it as reference evidence."""

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_prepare import PreparedMoE
from .backbone_runtime import open_backbone
from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION
from .generation import GenerationRequest, TextGeneration
from .generation_assets import GenerationAssets, GenerationMoE, reserve_disk
from .model_artifacts import canonical
from .output_assets import load as load_output
from .resources import GiB, MiB, MemoryBudget
from .tokenizer import PinnedTokenizer


def prepare(root, run, request, progress=None):
    root, run = Path(root), Path(run)
    if not run.resolve().is_relative_to((root / "generation").resolve()): raise ValueError("Expected a separate generation artifact directory")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "plan.json").exists(): raise FileExistsError("Refusing to replace a sealed generation plan")
    reserve_disk(run, 16 * GiB)
    bindings = json.loads((root / "backbone-controls.json").read_bytes())
    with ExitStack() as stack:
        budget = MemoryBudget(cap=12 * GiB, owner="generation_planning_fixtures"); budget.acquire(budget.cap)
        stack.callback(budget.release, budget.cap)
        row_budget = MemoryBudget(cap=16 * MiB, owner="generation_planning_rows"); row_budget.acquire(row_budget.cap)
        stack.callback(row_budget.release, row_budget.cap)
        reader = stack.enter_context(CatalogReader(root / "import-source"))
        tokenizer = PinnedTokenizer(root / "import-source"); prompt = request.encode(tokenizer)
        assets = GenerationAssets(root, run, reader)
        for token in prompt: assets.token(token)
        arrays = load_bindings(root, bindings); head_arrays, head_manifest = load_output(root / "output-fixture")
        providers = {l: PreparedMoE(root, bindings, l) for l in LAYERS}
        transfers = stack.enter_context(ThreadPoolExecutor(max_workers=8))
        planners = {l: GenerationMoE(root, run, bindings, l, reader, progress, transfer_pool=transfers) for l in LAYERS}
        engine = open_backbone(stack, root, arrays, assets.token_rows, providers, head_arrays, engram_rows=assets.rows)
        def step(ids, *, sample):
            assets.token(int(ids[0, 0]))
            candidate = engine.transaction.history.prepare(ids, engine.position)
            assets.prepare_rows(candidate)
            return engine.transaction.step(ids, lambda c: engine._finish(engine._evaluate(c, ids,
                lambda layer, values: planners[layer].resolve(values)), sample=sample))
        session = TextGeneration(tokenizer, step, request); calls = []
        while not session.done:
            meta, result = session.advance()
            calls.append({**meta, "routes": {str(l): planners[l].decisions[-1] for l in LAYERS}})
            if progress: progress({"planned_token": meta, "continuation": session.summary()})
        assets_report = assets.freeze()
        experts = {str(l): planners[l].freeze() for l in LAYERS}
        report = {"schema_version": 1, "probe": "text_generation_artifact_plan", "status": "pass", "model": MODEL, "revision": REVISION,
            "request": {"prompt": request.prompt, "max_new_tokens": request.max_new_tokens, "eos_id": request.eos_id},
            "layers": list(LAYERS), "temperature": .7, "seed": 2026, "bindings_sha256": bindings["bindings_sha256"],
            "output_fixture": head_manifest, "tokenizer": engine.tokenizer_identity, "calls": calls, "completion": session.summary(),
            "assets": assets_report, "experts": experts, "transaction": engine.transaction.stats(), "reader": reader.stats(),
            "maximum_parallel_payload_transfers": 8, "maximum_parallel_expert_imports": 4,
            "artifact_preparation_fetch_between_stages": True, "independent_model_comparison_run": False,
            "original_reference_execution_run": False}
        report["plan_sha256"] = canonical(report)
        (run / "plan.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
        return report
