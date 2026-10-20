"""Lossless, separately scoped input rows and original-route expert imports."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np

from .backbone import LAYERS
from .backbone_prepare import PreparedMoE
from .checkpoint import MODEL, REVISION, LIMIT, fetch_expert_fixture, load_expert_fixture
from .connected import ConnectedMoE
from .engram_assets import MAX_ROWS, row_address, expected_row_address
from .formats import bf16_values
from .gate import GateProjection, load_gate_fixture
from .model_artifacts import CATALOG_SHA256, canonical, digest, load_catalog
from .resources import GiB, MiB, MemoryBudget
from .router import RouterProbe, load_router_fixture
from .storage import build_store
from .backbone_runtime import embeddings, validate_moe_manifests


def reserve_disk(root, required=GiB):
    if shutil.disk_usage(root).free < 64 * GiB + required:
        raise OSError("Generation preparation would consume the 64 GiB disk reserve")


def embedding_address(catalog, token):
    if type(token) is not int or not 0 <= token < 129280: raise ValueError("Invalid original embedding ID")
    t = catalog["embed.weight"]
    if (t["dtype"], t["shape"]) != ("BF16", [129280, 5120]): raise ValueError("Original embedding shape differs")
    return {"tensor": "embed.weight", "dtype": "BF16", "shard": t["shard"],
        "shard_byte_offset": t["shard_byte_offset"] + token * 10240, "bytes": 10240}


def checked_cache(path, source, fetch):
    size = sum(s["bytes"] for s in source) if isinstance(source, list) else source["bytes"]
    meta = path.with_suffix(".json")
    if path.exists() or meta.exists():
        if not path.is_file() or path.stat().st_size != size or not meta.is_file() or meta.stat().st_size > 4096:
            raise ValueError("Incomplete generation row cache")
        m = json.loads(meta.read_bytes()); raw = path.read_bytes()
        if m.get("source") != source or m.get("sha256") != digest(raw): raise ValueError("Cached original generation row differs")
    else:
        raw = fetch(); m = {"source": source, "sha256": digest(raw)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file: file.write(raw)
        meta.write_bytes((json.dumps(m) + "\n").encode())
    if len(raw) != size: raise ValueError("Original generation row extent differs")
    return raw, m


@contextmanager
def prefetch_expert(reader, layer, expert, pool):
    """Overlap the same six bounded original payload reads; retain all checks.

    Four admitted expert owners share at most eight transfer workers. Their
    existing 128 MiB allowances include these 18,800,640 prefetched raw bytes.
    The unchanged importer still validates original headers, layout and hashes.
    """
    keys = []
    for projection in ("w1", "w2", "w3"):
        for suffix in ("weight", "scale"):
            t = reader.catalog[f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"]
            if not 0 < t["bytes"] <= LIMIT: raise ValueError("Unbounded original expert payload range")
            keys.append((t["shard"], t["shard_byte_offset"], t["bytes"]))
    if len(set(keys)) != 6 or sum(k[2] for k in keys) != 18800640: raise ValueError("Original expert prefetch extent differs")
    pending = {}
    class Reader:
        def small(self, path): return reader.small(path)
        def range(self, shard, start, size):
            key = shard, start, size
            if key in pending: return pending[key].result()
            if start in (0, 8): return reader.range(shard, start, size)
            raise ValueError("Importer requested an unprepared expert payload range")
    try:
        for key in keys: pending[key] = pool.submit(reader.range, *key)
        yield Reader()
    finally:
        # Keep ownership charged until every running transfer has returned,
        # including when a header or one transfer fails. Preserve the error
        # from the unchanged importer; queued unused requests can be cancelled.
        for future in pending.values(): future.cancel()
        for future in pending.values():
            try: future.result()
            except BaseException: pass


class PlanningRows:
    """Prepared host bytes only. Fetching is explicit and precedes model steps."""
    def __init__(self, run, reader, layer, *, max_rows=MAX_ROWS):
        from .sequence import MAX_ROWS as sequence_rows
        if type(max_rows) is not int or max_rows not in (MAX_ROWS, sequence_rows): raise ValueError("Unknown explicit selected-row profile")
        self.run, self.reader, self.layer = Path(run), reader, layer
        self.max_rows = max_rows
        self.entries, self.manifest = {}, {}

    def prepare(self, ids):
        requested = set(map(int, ids.reshape(-1)))
        if len(set(self.entries) | requested) > self.max_rows: raise ValueError("Generation Engram fixture row limit exceeded")
        def fetch(row):
            source = row_address(self.reader.catalog, self.layer, row)
            path = self.run / f"row-cache/{self.layer}-{row}.bin"
            raw, m = checked_cache(path, source, lambda: b"".join(self.reader.range(e["shard"], e["shard_byte_offset"], e["bytes"]) for e in source))
            if any((b & 127) == 127 for b in raw[:256]) or 255 in raw[256:]: raise ValueError("Nonfinite original Engram row")
            return row, (raw, {**m, "weight_sha256": digest(raw[:256]), "scale_sha256": digest(raw[256:])})
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row, entry in pool.map(fetch, sorted(requested - self.entries.keys())): self.entries[row] = entry

    def get(self, ids):
        if any(int(i) not in self.entries for i in ids.reshape(-1)): raise ValueError("Prepare original Engram rows before execution")
        raw = [self.entries[int(i)][0] for i in ids.reshape(-1)]
        return (np.frombuffer(b"".join(r[:256] for r in raw), np.uint8).reshape(*ids.shape, 256).copy(),
            np.frombuffer(b"".join(r[256:] for r in raw), np.uint8).reshape(*ids.shape, 8).copy())

    def freeze(self):
        directory = self.run / f"rows/layer{self.layer}"
        if directory.exists(): raise FileExistsError("Refusing to replace a sealed generation Engram store")
        directory.mkdir(parents=True); entries = []
        with (directory / "rows.bin").open("xb", buffering=0) as file:
            for n, (row, (raw, meta)) in enumerate(sorted(self.entries.items())):
                if file.write(raw + bytes(512 - 264)) != 512: raise OSError("Short generation Engram store write")
                entries.append({"row": row, "offset": n * 512, **meta})
            padding = (-file.tell()) % 4096
            if padding: file.write(bytes(padding))
            size = file.tell(); os.fsync(file.fileno())
        m = {"kind": "original_engram_selected_rows", "model": MODEL, "revision": REVISION, "catalog_sha256": CATALOG_SHA256,
            "layer": self.layer, "rows": entries, "record_bytes": 512, "file_bytes": size, "original_payload_bytes": len(entries) * 264,
            "logical_rows_per_record": 1, "physical_alignment_bytes": 4096, "maximum_rows": self.max_rows}
        (directory / "manifest.json").write_bytes((json.dumps(m, indent=2) + "\n").encode())
        self.manifest = m
        return m


class GenerationAssets:
    def __init__(self, root, run, reader, *, max_embeddings=8, max_rows=MAX_ROWS):
        from .sequence import MAX_EMBEDDINGS
        if type(max_embeddings) is not int or max_embeddings not in (8, MAX_EMBEDDINGS): raise ValueError("Unknown explicit embedding profile")
        self.root, self.run, self.reader = Path(root), Path(run), reader
        self.max_embeddings = max_embeddings
        self.token_rows, self.embedding_receipts = {}, {}
        self.reuse = embeddings(root)
        self.rows = {l: PlanningRows(run, reader, l, max_rows=max_rows) for l in (1, 14)}

    def token(self, token):
        if token in self.token_rows: return
        if len(self.token_rows) >= self.max_embeddings: raise ValueError("Generation embedding fixture limit")
        source = embedding_address(self.reader.catalog, token)
        path = self.run / f"embeddings/{token}.bin"
        raw, m = checked_cache(path, source, lambda: self.reuse[token].tobytes() if token in self.reuse else
            self.reader.range(source["shard"], source["shard_byte_offset"], source["bytes"]))
        a = np.frombuffer(raw, np.uint16).copy()
        if a.shape != (5120,) or np.any((a & 0x7f80) == 0x7f80): raise ValueError("Invalid finite original embedding row")
        a.flags.writeable = False
        self.token_rows[token] = a
        self.embedding_receipts[token] = {"id": token, "file": path.relative_to(self.run).as_posix(), **m}

    def prepare_rows(self, candidate):
        for i, layer in enumerate((1, 14)): self.rows[layer].prepare(candidate.hash_ids[:, :, i, :])

    def freeze(self):
        return {"embeddings": [self.embedding_receipts[i] for i in sorted(self.embedding_receipts)],
            "Engram_rows": {str(l): rows.freeze() for l, rows in self.rows.items()}}


class GenerationMoE(PreparedMoE):
    """Planning-only importer with one temporary selected-route store per call.

    Immutable final stores are built once, after tracing, avoiding quadratic
    copies of growing unions. Reused checked NPZ files are hard-linked when the
    filesystem supports it. Public execution uses ordinary offline PreparedMoE.
    """
    def __init__(self, root, run, bindings, layer, reader, progress=None, *, transfer_pool=None):
        super().__init__(root, bindings, layer)
        self.run, self.reader, self.progress = Path(run), reader, progress
        self.directory = self.run / f"experts/layer{layer}"
        self.paths, self.decisions, self.fetched = {}, [], 0
        self.transfer_pool = transfer_pool

    def fetch(self, expert):
        reserve_disk(self.run)
        budget = MemoryBudget(cap=128 * MiB); budget.acquire(budget.cap)
        try:
            folder = self.directory / f"expert-{expert:03d}"; path = folder / "native-expert.npz"
            count = 0
            if not path.exists():
                reuse = self.root / f"backbone-experts/layer{self.layer}/{folder.name}"
                if (reuse / path.name).exists():
                    arrays, m = load_expert_fixture(reuse / path.name); del arrays
                    if (m["layer"], m["expert"]) != (self.layer, expert): raise ValueError("Reused expert identity differs")
                    folder.mkdir(parents=True, exist_ok=True)
                    for name in (path.name, "manifest.json"):
                        try: os.link(reuse / name, folder / name)
                        except OSError: shutil.copyfile(reuse / name, folder / name)
                else:
                    if self.transfer_pool is None:
                        m = fetch_expert_fixture(folder, self.layer, expert, reader=self.reader)
                    else:
                        with prefetch_expert(self.reader, self.layer, expert, self.transfer_pool) as reader:
                            m = fetch_expert_fixture(folder, self.layer, expert, reader=reader)
                    count = m["weights_fetched_bytes"]
            arrays, m = load_expert_fixture(path); del arrays
            if (m["layer"], m["expert"]) != (self.layer, expert): raise ValueError("Prepared generation expert identity differs")
            return expert, path, count
        finally: budget.release(budget.cap)

    def resolve(self, values):
        weight, _ = load_gate_fixture(self.files[0]); bias, _ = load_router_fixture(self.files[1])
        with GateProjection(weight) as gate, RouterProbe(bias) as router:
            route = router.inspect(gate.project(bf16_values(values)), False)
        ids = set(map(int, route["indices"]))
        if len(ids | self.available) > 48: raise ValueError("Generation fixture exceeds the explicit 48-expert store limit")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for expert, path, count in pool.map(self.fetch, sorted(ids - self.paths.keys())):
                self.paths[expert] = path; self.fetched += count
        scratch = self.run / "scratch"; scratch.mkdir(exist_ok=True)
        # TemporaryDirectory owns exactly its newly created directory, within
        # this run's scratch path. Historical qualified stores are never moved.
        with tempfile.TemporaryDirectory(prefix="route-", dir=scratch) as temporary:
            target = Path(temporary) / "store"
            if not target.resolve().is_relative_to(scratch.resolve()): raise ValueError("Unbounded temporary store path")
            build_store([self.paths[e] for e in sorted(ids)], target, max_experts=6)
            with ConnectedMoE(target, *self.files, layer=self.layer, profile="backbone_probe") as moe:
                validate_moe_manifests(moe, self.bindings, self.layer)
                result = moe.forward(values, False)
        if (not np.array_equal(result["routing"]["indices"], route["indices"])
                or not np.array_equal(result["routing"]["weights"].view(np.uint32), route["weights"].view(np.uint32))):
            raise RuntimeError("Generation preparation changed the original route")
        self.available |= ids
        self.decisions.append({"indices": route["indices"].tolist(), "weight_bits": route["weights"].view(np.uint32).tolist()})
        if self.progress: self.progress({"prepared_layer": self.layer, "position": len(self.decisions) - 1, "experts": len(self.available)})
        return result

    def freeze(self):
        reserve_disk(self.run, len(self.paths) * 18800640 + MiB)
        target = self.run / f"stores/layer{self.layer}/{canonical(sorted(self.available))[:20]}"
        build_store([self.paths[e] for e in sorted(self.paths)], target, max_experts=48)
        return {"ids": sorted(self.available), "store": target.relative_to(self.root).as_posix(), "new_original_bytes": self.fetched}


def load_token_rows(root, run, plan):
    catalog, _ = load_catalog(Path(root) / "import-source")
    entries = plan["assets"]["embeddings"]
    if not 1 <= len(entries) <= 8 or [e["id"] for e in entries] != sorted(set(e["id"] for e in entries)):
        raise ValueError("Invalid generation embedding fixture census")
    rows = {}
    for entry in entries:
        token = entry["id"]; relative = Path(entry["file"])
        if relative.as_posix() != f"embeddings/{token}.bin" or entry["source"] != embedding_address(catalog, token):
            raise ValueError("Original embedding source address differs")
        path = Path(run) / relative
        if path.stat().st_size != 10240: raise ValueError("Original embedding row byte count differs")
        raw = path.read_bytes()
        if len(raw) != 10240 or digest(raw) != entry["sha256"]: raise ValueError("Original embedding row payload differs")
        a = np.frombuffer(raw, np.uint16)
        if np.any((a & 0x7f80) == 0x7f80): raise ValueError("Nonfinite prepared original embedding")
        rows[token] = a
    for layer in (1, 14):
        path = Path(run) / f"rows/layer{layer}/manifest.json"
        if path.stat().st_size > MiB: raise ValueError("Excessive generation Engram metadata")
        m = json.loads(path.read_bytes())
        if m != plan["assets"]["Engram_rows"][str(layer)] or any(r["source"] != expected_row_address(layer, r["row"]) for r in m["rows"]):
            raise ValueError("Generation Engram source binding differs")
    return rows
