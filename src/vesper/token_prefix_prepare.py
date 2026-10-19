"""Prepare the measured prefix's original experts outside the forward path."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
import shutil

import numpy as np

from .attention_validation import array_digest
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .connected import ConnectedMoE
from .formats import bf16_values
from .gate import GateProjection, load_gate_fixture
from .router import RouterProbe, load_router_fixture
from .storage import build_store
from .token_prefix import LAYERS, TOKENS
from .token_prefix_runtime import load_arrays, moe_files, open_prefix
from .resources import MemoryBudget, GiB


class MissingPrefixExperts(KeyError):
    def __init__(self, layer, selected):
        self.layer, self.selected = layer, selected
        super().__init__("Prepare required original experts outside the token computation")


class PlanningMoE:
    def __init__(self, stack, root, layer, ordinal=0):
        self.root, self.layer, self.ordinal = Path(root), layer, ordinal
        self.files = moe_files(root, layer)
        weight, self.gate_manifest = load_gate_fixture(self.files[0])
        bias, self.router_manifest = load_router_fixture(self.files[1])
        self.gate = stack.enter_context(GateProjection(weight, ordinal))
        self.router = stack.enter_context(RouterProbe(bias, ordinal))
        self.available, self.store = set(), None
        self.decisions = []

    def forward(self, values, image_span=False):
        if image_span:
            raise ValueError("The prefix preparation corpus is text-only")
        routing = self.router.inspect(self.gate.project(bf16_values(values)), False)
        selected = list(map(int, routing["indices"]))
        if not set(selected).issubset(self.available):
            raise MissingPrefixExperts(self.layer, selected)
        with ConnectedMoE(self.store, *self.files, layer=self.layer, profile="token_prefix") as moe:
            result = moe.forward(values, False)
        if (result["routing"]["indices"].tolist() != selected
                or not np.array_equal(result["routing"]["weights"], routing["weights"])):
            raise RuntimeError("Prepared native routing differs from the live original gate")
        self.decisions.append({"input_bf16_sha256": array_digest(values), "indices": selected,
            "weight_bits": routing["weights"].view(np.uint32).tolist()})
        return result

    def prepare(self, selected, progress=None):
        requested = self.available | set(selected)
        if len(requested) > 12:
            raise MemoryError("Two-token prefix preparation exceeds twelve original experts per layer")
        root = self.root / f"token-prefix-experts/layer{self.layer}"
        reuse = self.root / ("connected-experts" if self.layer == 0 else f"block{self.layer}-experts")
        def fetch(expert):
            folder = root / f"expert-{expert:03d}"
            path = folder / "native-expert.npz"
            existing = reuse / folder.name / path.name
            fetched = 0
            if not path.exists():
                if existing.exists():
                    # Copy only a checked original fixture, never an unlabelled
                    # cache payload. Copying is bounded by the file API.
                    arrays, m = load_expert_fixture(existing); del arrays
                    if (m["layer"], m["expert"]) != (self.layer, expert):
                        raise ValueError("Reused prefix expert identity differs")
                    folder.mkdir(parents=True, exist_ok=False)
                    shutil.copyfile(existing, path)
                    shutil.copyfile(existing.parent / "manifest.json", folder / "manifest.json")
                else:
                    m = fetch_expert_fixture(folder, layer=self.layer, expert=expert)
                    fetched = m["weights_fetched_bytes"]
            arrays, m = load_expert_fixture(path); del arrays
            if (m["layer"], m["expert"]) != (self.layer, expert):
                raise ValueError("Prepared prefix expert belongs to another original layer")
            return path, fetched
        paths, fetched = [], 0
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(fetch, e) for e in sorted(requested)]):
                path, size = future.result(); paths.append(path); fetched += size
        target = self.root / f"token-prefix-stores/layer{self.layer}/experts-{len(requested)}"
        if not target.exists():
            build_store(sorted(paths), target, max_experts=48)
        self.store, self.available = target, requested
        if progress:
            progress({"prepared_layer": self.layer, "experts": len(requested), "new_original_bytes": fetched})


def prepare(root, progress=None):
    root = Path(root)
    # Fixture arrays and fetch/store diagnostics are accounted separately from
    # the opened arithmetic components and their bounded state snapshots.
    budget = MemoryBudget(cap=2 * GiB); budget.acquire(budget.cap)
    try:
        arrays, embeddings, manifests = load_arrays(root)
        with ExitStack() as stack:
            moe = {l: PlanningMoE(stack, root, l) for l in LAYERS}
            prefix = open_prefix(stack, root, arrays, embeddings, moe)
            calls = []
            for token in TOKENS:
                retries = 0
                while True:
                    try:
                        result = prefix.step(np.array([[token]], np.int64))
                        calls.append({"token_id": token, "position": prefix.position - 1,
                            "output_bf16_sha256": array_digest(result["output_bf16"]),
                            "next_pre_mix_fp32_sha256": array_digest(result["next_pre_mix_fp32"]),
                            "decisions": {str(l): moe[l].decisions[-1] for l in LAYERS}})
                        break
                    except MissingPrefixExperts as missing:
                        if prefix.transaction.pending:
                            raise RuntimeError("Never prepare artifacts while token recovery is pending")
                        retries += 1
                        if retries > 4:
                            raise RuntimeError("Preparation revisited a route without completing its original expert store")
                        moe[missing.layer].prepare(missing.selected, progress)
            report = {"schema_version": 1, "probe": "original_connected_token_prefix_expert_plan", "status": "pass",
                "model": MODEL, "revision": REVISION, "layers": list(LAYERS), "tokens": list(TOKENS),
                "fixtures": manifests, "additional_identities": prefix.probe_identities, "calls": calls,
                "experts": {str(l): {"ids": sorted(moe[l].available), "store": str(moe[l].store.relative_to(root)).replace('\\', '/'),
                    "gate": moe[l].gate_manifest, "router": moe[l].router_manifest} for l in LAYERS},
                "transaction": prefix.transaction.stats(), "full_model_execution_run": False,
                "text_generation_available": False, "weight_fetch_inside_forward": False}
        return report
    finally:
        budget.release(budget.cap)
