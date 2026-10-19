"""Bounded original-route artifact planning, separate from executable inference."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import shutil

import numpy as np

from .attention_validation import array_digest
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_runtime import embeddings, moe_files, open_backbone, validate_moe_manifests
from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .connected import ConnectedMoE
from .formats import bf16_values
from .gate import GateProjection, load_gate_fixture
from .model_artifacts import canonical
from .output_assets import load as load_output
from .resources import GiB, MemoryBudget
from .router import RouterProbe, load_router_fixture
from .storage import build_store

TOKENS = (0, 2)


class PreparedMoE:
    """Open one layer's immutable weights per call; never fetch in forward."""

    def __init__(self, root, bindings, layer, prepared=None, expected_provenance=None):
        self.root, self.layer = Path(root), layer
        self.files = moe_files(root, bindings, layer)
        self.bindings, self.expected_provenance = bindings, expected_provenance
        self.available, self.store = set(), None
        if prepared is not None:
            self.available = set(prepared["ids"])
            relative = Path(prepared["store"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Expected a relative immutable original expert store")
            self.store = self.root / relative

    def forward(self, values, image_span=False):
        if self.store is None: raise RuntimeError("Prepare original experts before inference")
        with ConnectedMoE(self.store, *self.files, layer=self.layer, profile="backbone_probe") as moe:
            validate_moe_manifests(moe, self.bindings, self.layer)
            if self.expected_provenance is not None:
                entries = moe.experts.store.entries
                if set(entries) != {f"{self.layer}:{e}" for e in self.available} or any(
                        e["provenance"] != self.expected_provenance[k] for k, e in entries.items()):
                    raise ValueError("Native original expert payload provenance differs")
            return moe.forward(values, image_span)

    def install(self, selected, reader, progress=None):
        if shutil.disk_usage(self.root).free < 65 * GiB:
            raise OSError("Expert preparation would consume the 64 GiB disk reserve")
        requested = self.available | set(map(int, selected))
        if len(requested) > 12 or any(not 0 <= e < 384 for e in requested):
            raise ValueError("Two-token planning is limited to twelve original experts per layer")
        directory = self.root / f"backbone-experts/layer{self.layer}"
        reuse = [self.root / f"token-prefix-experts/layer{self.layer}",
            self.root / ("connected-experts" if self.layer == 0 else f"block{self.layer}-experts")]
        def fetch(expert):
            budget = MemoryBudget(cap=128 * 1024**2); budget.acquire(budget.cap)
            try:
                folder = directory / f"expert-{expert:03d}"
                path = folder / "native-expert.npz"
                fetched = 0
                if not path.exists():
                    existing = next((r / folder.name / path.name for r in reuse if (r / folder.name / path.name).exists()), None)
                    if existing is not None:
                        arrays, manifest = load_expert_fixture(existing); del arrays
                        if (manifest["layer"], manifest["expert"]) != (self.layer, expert):
                            raise ValueError("Reused expert belongs to another original layer")
                        folder.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(existing, path)
                        shutil.copyfile(existing.parent / "manifest.json", folder / "manifest.json")
                    else:
                        manifest = fetch_expert_fixture(folder, self.layer, expert, reader=reader)
                        fetched = manifest["weights_fetched_bytes"]
                arrays, manifest = load_expert_fixture(path); del arrays
                if (manifest["layer"], manifest["expert"]) != (self.layer, expert):
                    raise ValueError("Prepared expert belongs to another original layer")
                return path, fetched
            finally: budget.release(budget.cap)
        paths, fetched = [], 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(fetch, e) for e in sorted(requested)]):
                path, size = future.result(); paths.append(path); fetched += size
        identity = canonical({"layer": self.layer, "ids": sorted(requested)})[:20]
        target = self.root / f"backbone-stores/layer{self.layer}/{identity}"
        if not target.exists(): build_store(sorted(paths), target, max_experts=48)
        self.store, self.available = target, requested
        if progress: progress({"prepared_layer": self.layer, "experts": len(requested), "new_original_bytes": fetched})


def prepare(root, bindings, progress=None):
    """Trace two original routes; the special planner may fetch between stages.

    Public NativeBackbone.step has no resolver or fetch path. This operation is
    artifact preparation and is not counted as independent execution evidence.
    """
    root = Path(root)
    if shutil.disk_usage(root).free < 80 * GiB: raise OSError("Keep 64 GiB free plus bounded expert preparation space")
    with ExitStack() as lifetime:
        budget = MemoryBudget(cap=12 * GiB); budget.acquire(budget.cap)
        lifetime.callback(budget.release, budget.cap)
        reader = lifetime.enter_context(CatalogReader(root / "import-source"))
        arrays = load_bindings(root, bindings)
        checked = bindings
        head_arrays, head_manifest = load_output(root / "output-fixture")
        providers = {l: PreparedMoE(root, checked, l) for l in LAYERS}
        decisions = {l: [] for l in LAYERS}
        with ExitStack() as stack:
            engine = open_backbone(stack, root, arrays, embeddings(root), providers, head_arrays)
            def resolve(layer, values):
                provider = providers[layer]
                weight, _ = load_gate_fixture(provider.files[0]); bias, _ = load_router_fixture(provider.files[1])
                with GateProjection(weight) as gate, RouterProbe(bias) as router:
                    route = router.inspect(gate.project(bf16_values(values)), False)
                if not set(map(int, route["indices"])).issubset(provider.available):
                    provider.install(route["indices"], reader, progress)
                result = provider.forward(values, False)
                if (not np.array_equal(result["routing"]["indices"], route["indices"])
                        or not np.array_equal(result["routing"]["weights"].view(np.uint32), route["weights"].view(np.uint32))):
                    raise RuntimeError("Prepared route differs from the unchanged native gate")
                decisions[layer].append({"input_bf16_sha256": array_digest(values), "indices": route["indices"].tolist(),
                    "weight_bits": route["weights"].view(np.uint32).tolist()})
                return result
            calls = []
            for token in TOKENS:
                ids = np.array([[token]], np.int64)
                result = engine.transaction.step(ids, lambda c: engine._finish(engine._evaluate(c, ids, resolve)))
                calls.append({"token_id": token, "position": engine.position - 1,
                    "output_bf16_sha256": array_digest(result["output_bf16"]),
                    "next_pre_mix_fp32_sha256": array_digest(result["next_pre_mix_fp32"]),
                    "logits_fp32_sha256": array_digest(result["head"]["logits_fp32"]),
                    "sampled_id": result["sampling"]["ids_i64"].tolist(),
                    "decisions": {str(l): decisions[l][-1] for l in LAYERS}})
            report = {"schema_version": 1, "probe": "original_backbone_artifact_plan", "status": "pass",
                "model": MODEL, "revision": REVISION, "layers": list(LAYERS), "tokens": list(TOKENS),
                "bindings_sha256": checked["bindings_sha256"], "head": head_manifest, "calls": calls,
                "experts": {str(l): {"ids": sorted(p.available), "store": p.store.relative_to(root).as_posix()} for l, p in providers.items()},
                "tokenizer": engine.tokenizer_identity, "transaction": engine.transaction.stats(),
                "artifact_preparation_fetch_between_stages": True, "public_inference_fetch_available": False,
                "independent_model_comparison_run": False, "text_generation_available": False, "reader": reader.stats()}
        return report
