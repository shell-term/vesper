"""Pinned Spark acquisition and offline reads of unchanged original experts.

Full acquisition is explicit. Files are streamed into one resumable partial,
verified against the released whole-file SHA-256, and exclusively published.
The expert adapter gathers six original tensor spans into caller-owned staging;
it does not download, requantize, or create a second model-wide expert pack.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import urllib.request

import numpy as np

from .artifact_repository import VerifiedBlob, register_existing
from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, BASE, EXPERT_LAYOUT, validate_expert_identity
from .integrity import digest_shape, hash_chunks
from .model_artifacts import CATALOG_SHA256, SHARD_INVENTORY_SHA256, DTYPE_BYTES, FILES, load_catalog, shard_inventory, digest
from .resources import MemoryBudget, MiB, GiB
from .storage import tensor_layout
from .timing import Timings

CHUNK_BYTES = 8 * MiB
MIN_CHUNK_BYTES = 2 * MiB
MAX_JOURNAL_BYTES = 8 * MiB
DOWNLOAD_METADATA_ALLOWANCE = 64 * MiB
EXPERT_METADATA_BYTES = 16 * MiB


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _scope(workspace, *paths):
    if workspace is None:
        from .workspace import Workspace
        workspace = Workspace.open()
    workspace.check()
    context = workspace.context()
    if context["platform"] != "dgx-spark" or context["profile"] != "native":
        raise ValueError("Original acquisition requires a dgx-spark/native workspace")
    if context.get("model") != MODEL or context.get("revision") != REVISION:
        raise ValueError("Original acquisition requires the pinned original model and revision")
    workspace.check_runtime_module(__file__, "src/vesper/spark_artifacts.py")
    from . import workspace as workspace_module
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    base = workspace.artifact_path("namespace-check").parent.resolve()
    for value in paths:
        path = Path(value).absolute()
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as error:
            raise ValueError("Original artifact paths must stay in this Spark artifact namespace") from error
        if workspace.artifact_path(relative).absolute() != path:
            raise ValueError("Original artifact path differs from its bound namespace")
    return workspace


def _validate_chunk(chunk_bytes):
    if type(chunk_bytes) is not int or not MIN_CHUNK_BYTES <= chunk_bytes <= CHUNK_BYTES or chunk_bytes % 65536:
        raise ValueError("Download chunks must be aligned, between 2 and 8 MiB")


def _identity(expected, chunk_bytes):
    if (not re.fullmatch(r"model-\d{5}-of-00048\.safetensors", expected.get("file", ""))
            or type(expected.get("bytes")) is not int or expected["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", ""))):
        raise ValueError("Invalid released original shard identity")
    return {"kind": "vesper_original_partial_v1", "model": MODEL, "revision": REVISION,
            "expected": expected, "chunk_bytes": chunk_bytes}


def _existing_bytes(root, inventory):
    allowed = {".download.lock", "download-manifest.json"}
    for row in inventory:
        allowed.update((row["file"], row["file"] + ".part", row["file"] + ".chunks.jsonl"))
    total = 0
    if root.exists():
        for child in root.iterdir():
            if child.is_symlink() or not child.is_file() or child.name not in allowed:
                raise ValueError("Unexpected file or link in the dedicated original download directory")
            if (child.name.endswith((".part", ".chunks.jsonl")) or child.name == ".download.lock") and child.stat().st_nlink != 1:
                raise ValueError("Writable original partials, journals and locks must have one owned link")
            total += child.stat().st_size
    return total


def plan_originals(metadata, originals, *, quota_bytes, disk_reserve=64 * GiB,
                   chunk_bytes=CHUNK_BYTES, workspace=None):
    """Read-only resource admission; does not open a network connection."""
    _validate_chunk(chunk_bytes)
    _scope(workspace, metadata, originals)
    _regular_below(Path(metadata), Path(metadata) / "tree.json")
    if type(disk_reserve) is not int or disk_reserve < 32 * GiB:
        raise ValueError("Keep at least a 32 GiB SSD reserve")
    inventory = shard_inventory(metadata)
    expected = sum(row["bytes"] for row in inventory)
    required = expected + DOWNLOAD_METADATA_ALLOWANCE
    if type(quota_bytes) is not int or quota_bytes < required:
        raise ValueError("Declared original-download quota is insufficient")
    root = Path(originals)
    occupied = _existing_bytes(root, inventory)
    if occupied > quota_bytes:
        raise ValueError("Existing original download directory already exceeds its quota")
    if any(((row["bytes"] + chunk_bytes - 1) // chunk_bytes) * 192 + 4096 > MAX_JOURNAL_BYTES for row in inventory):
        raise ValueError("Chunk receipts would exceed their metadata bound")
    parent = root
    while not parent.exists():
        parent = parent.parent
    remaining = max(0, required - occupied)
    free = shutil.disk_usage(parent).free
    if free < remaining + disk_reserve:
        raise OSError("Original acquisition would consume the SSD reserve")
    return {"kind": "original_download_plan", "model": MODEL, "revision": REVISION,
            "shards": len(inventory), "original_bytes": expected, "quota_bytes": quota_bytes,
            "metadata_allowance_bytes": DOWNLOAD_METADATA_ALLOWANCE, "existing_bytes": occupied,
            "remaining_capacity_admission_bytes": remaining, "disk_reserve_bytes": disk_reserve,
            "chunk_bytes": chunk_bytes, "whole_checkpoint_downloaded": False,
            "network_started": False, "second_expert_pack_created": False}


@contextmanager
def _download_lock(root):
    with (root / ".download.lock").open("a+b") as file:
        file.seek(0, 2)
        if not file.tell():
            file.write(b"\0"); file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _hash_file(path, chunk_bytes, checkpoint=lambda: None):
    sha = hashlib.sha256()
    with path.open("rb", buffering=0) as file:
        while raw := file.read(chunk_bytes):
            checkpoint()
            sha.update(raw)
    return sha.hexdigest()


def _write_all(file, raw):
    view = memoryview(raw)
    while view:
        written = file.write(view)
        if not written:
            raise OSError("Short original artifact write")
        view = view[written:]


def _resume(part, journal, identity, checkpoint=lambda: None):
    """Recheck every acknowledged byte; discard only owned unacknowledged tails."""
    sha, offset = hashlib.sha256(), 0
    expected, chunk = identity["expected"], identity["chunk_bytes"]
    if journal.is_symlink() or part.is_symlink():
        raise ValueError("Original partials cannot be symlinks")
    if not journal.exists():
        if part.exists():
            raise ValueError("Unregistered original partial requires inspection")
        with journal.open("xb", buffering=0) as file:
            _write_all(file, _json(identity)); os.fsync(file.fileno())
    if journal.stat().st_size > MAX_JOURNAL_BYTES:
        raise ValueError("Excessive original partial metadata")
    with journal.open("r+b", buffering=0) as records:
        first = records.readline(4097)
        if len(first) > 4096 or not first.endswith(b"\n") or json.loads(first) != identity:
            raise ValueError("Original partial source, revision or chunk size differs")
        if not part.exists():
            if records.read(1):
                raise ValueError("Acknowledged original partial data is missing")
            with part.open("xb"):
                pass
            records.seek(len(first))
        with part.open("r+b", buffering=0) as data:
            while True:
                checkpoint()
                position = records.tell()
                line = records.readline(1025)
                if not line:
                    break
                if len(line) > 1024:
                    raise ValueError("Excessive original chunk receipt")
                if not line.endswith(b"\n"):
                    if records.read(1):
                        raise ValueError("Invalid original chunk receipt")
                    records.truncate(position); os.fsync(records.fileno()); break
                row = json.loads(line)
                size = min(chunk, expected["bytes"] - offset)
                if size <= 0 or row.get("offset") != offset or row.get("bytes") != size:
                    raise ValueError("Original partial chunk extent differs")
                raw = data.read(size)
                if len(raw) != size or row.get("sha256") != digest(raw):
                    raise ValueError("Original partial acknowledged bytes changed")
                sha.update(raw); offset += size
            if data.seek(0, 2) < offset:
                raise ValueError("Original partial is shorter than acknowledged bytes")
            data.truncate(offset); os.fsync(data.fileno())
    return offset, sha


def _response_extent(response, start, expected):
    size = expected["bytes"]
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Compressed original transfer is not an exact byte stream")
    remaining = size - start
    if response.status == 206:
        if response.headers.get("Content-Range") != f"bytes {start}-{size - 1}/{size}":
            raise ValueError("Original transfer range or total extent differs")
    elif response.status != 200 or start != 0:
        raise ValueError("Server ignored a resumed original byte range")
    length = response.headers.get("Content-Length")
    if length is None or not length.isdecimal() or int(length) != remaining:
        raise ValueError("Original transfer content length differs")


def _publish(part, final):
    if part.parent != final.parent:
        raise ValueError("Original publication must stay within its owned directory")
    if os.name == "nt":
        # Windows rename is atomic and refuses an existing destination.
        os.rename(part, final)
    else:
        # POSIX rename can overwrite a raced destination; hard-link publication
        # is exclusive, atomic and creates no second payload copy.
        os.link(part, final)
        part.unlink()
        directory = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _download_one(root, expected, chunk_bytes, *, opener, checkpoint, progress=None):
    identity = _identity(expected, chunk_bytes)
    final = root / expected["file"]
    part, journal = root / (final.name + ".part"), root / (final.name + ".chunks.jsonl")
    if final.is_symlink() or part.is_symlink() or journal.is_symlink():
        raise ValueError("Original downloads cannot replace or follow links")
    if final.exists():
        if part.exists() or final.stat().st_size != expected["bytes"] or _hash_file(final, chunk_bytes, checkpoint) != expected["sha256"]:
            raise ValueError("Existing original file differs or has an ambiguous partial")
        return 0
    offset, sha = _resume(part, journal, identity, checkpoint)
    received = 0
    if offset < expected["bytes"]:
        request = urllib.request.Request(f"{BASE}/resolve/{REVISION}/{final.name}",
            headers={"Range": f"bytes={offset}-{expected['bytes'] - 1}", "Accept-Encoding": "identity"})
        with opener(request, timeout=45) as response:
            _response_extent(response, offset, expected)
            with part.open("r+b", buffering=0) as output, journal.open("ab", buffering=0) as records:
                output.seek(offset)
                while offset < expected["bytes"]:
                    checkpoint()
                    wanted = min(chunk_bytes, expected["bytes"] - offset)
                    raw = bytearray()
                    while len(raw) < wanted:
                        block = response.read(wanted - len(raw))
                        if not block:
                            raise OSError("Truncated original transfer; acknowledged prefix retained")
                        if len(block) > wanted - len(raw):
                            raise ValueError("Original transfer exceeded its bounded read request")
                        raw.extend(block)
                    _write_all(output, raw); os.fsync(output.fileno())
                    row = {"offset": offset, "bytes": len(raw), "sha256": digest(raw)}
                    _write_all(records, _json(row)); os.fsync(records.fileno())
                    sha.update(raw); offset += len(raw); received += len(raw)
                    if progress:
                        progress({"shard": final.name, "acknowledged_bytes": offset, "total_bytes": expected["bytes"]})
                if response.read(1):
                    raise ValueError("Original transfer exceeded its declared extent")
    if offset != expected["bytes"] or sha.hexdigest() != expected["sha256"]:
        raise ValueError("Original whole-file SHA-256 differs; partial preserved")
    checkpoint()
    _publish(part, final)
    return received


def download_originals(metadata, originals, *, quota_bytes, disk_reserve=64 * GiB,
                       chunk_bytes=CHUNK_BYTES, execute=False, workspace=None,
                       opener=None, progress=None):
    """Explicit full import, with one original copy and bounded transfer memory."""
    if execute is not True:
        raise ValueError("Full original download requires execute=True; inspect plan_originals first")
    ws = _scope(workspace, metadata, originals)
    initial_context = ws.context()
    plan = plan_originals(metadata, originals, quota_bytes=quota_bytes, disk_reserve=disk_reserve,
                          chunk_bytes=chunk_bytes, workspace=ws)
    inventory = shard_inventory(metadata)
    root = Path(originals); root.mkdir(parents=True, exist_ok=True)
    budget = MemoryBudget(cap=64 * MiB, owner="spark_original_download")
    budget.acquire(budget.cap)
    new_bytes = 0
    try:
        with _download_lock(root):
            def checkpoint():
                budget.acquire(0)
                if _existing_bytes(root, inventory) > quota_bytes:
                    raise OSError("Original acquisition exceeded its directory quota")
                if shutil.disk_usage(root).free < disk_reserve + DOWNLOAD_METADATA_ALLOWANCE:
                    raise OSError("Original acquisition reached the SSD reserve")
            for row in inventory:
                ws.check(); checkpoint()
                new_bytes += _download_one(root, row, chunk_bytes, opener=opener or urllib.request.urlopen,
                                            checkpoint=checkpoint, progress=progress)
            ws.check()
            ws.stamp_result({}, initial_context=initial_context)
            result = {"kind": "verified_original_download", "model": MODEL, "revision": REVISION,
                      "shard_inventory_sha256": SHARD_INVENTORY_SHA256,
                      "original_bytes": plan["original_bytes"], "shards": len(inventory),
                      "new_network_payload_bytes": new_bytes, "whole_checkpoint_downloaded": len(inventory) == 48,
                      "whole_shard_SHA256_verified": True, "quota_bytes": quota_bytes,
                      "disk_reserve_bytes": disk_reserve, "transfer_memory_allowance_bytes": budget.cap,
                      "second_expert_pack_created": False, "runtime_or_quality_qualified": False}
            manifest = root / "download-manifest.json"
            if manifest.exists():
                if not 0 < manifest.stat().st_size <= 65536:
                    raise ValueError("Invalid existing original completion metadata")
                old = json.loads(manifest.read_bytes())
                if any(old.get(k) != result[k] for k in ("kind", "model", "revision", "shard_inventory_sha256", "original_bytes", "shards", "whole_shard_SHA256_verified")):
                    raise ValueError("Existing original completion identity differs")
            else:
                with manifest.open("xb") as file:
                    file.write(_json(result)); file.flush(); os.fsync(file.fileno())
            return result
    finally:
        budget.release(budget.cap)


def register_originals(metadata, originals, integrity, *, execute=False, workspace=None):
    """Use the existing released-SHA registration and bounded integrity indexes."""
    _scope(workspace, metadata, originals, integrity)
    return register_existing(metadata, originals, integrity, execute=execute)


def _registration(metadata, integrity):
    path = Path(integrity) / "manifest.json"
    if not 0 < path.stat().st_size <= 65536:
        raise ValueError("Excessive original registration metadata")
    raw = path.read_bytes(); receipt = json.loads(raw)
    if (receipt.get("kind"), receipt.get("model"), receipt.get("revision"), receipt.get("catalog_sha256"), receipt.get("shard_inventory_sha256")) != (
            "registered_original_artifacts", MODEL, REVISION, CATALOG_SHA256, SHARD_INVENTORY_SHA256):
        raise ValueError("Original registration identity differs")
    inventory = {r["file"]: r for r in shard_inventory(metadata)}
    rows = receipt.get("entries", [])
    if len(rows) != 48 or {r["file"] for r in rows} != set(inventory):
        raise ValueError("All original shards must be registered before arbitrary offline routes")
    receipts = {r["file"]: r for r in rows}
    for name, entry in receipts.items():
        if any(entry.get(k) != v for k, v in inventory[name].items()):
            raise ValueError("Original registered shard identity differs")
    return inventory, receipts, digest(raw)


def _regular_below(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    if not path.is_relative_to(root):
        raise ValueError("Original reader file leaves its declared namespace")
    for node in (path, *path.parents):
        info = node.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Original reader files cannot follow links or reparse points")
        if node == root:
            break
    if not path.is_file():
        raise ValueError("Original reader requires a regular local file")


def _metadata_paths(metadata):
    root = Path(metadata)
    for filename in [*(value[0] for value in FILES.values()), "tree.json"]:
        _regular_below(root, root / filename)
    for ordinal in range(1, 49):
        _regular_below(root, root / "headers" / f"model-{ordinal:05d}-of-00048.safetensors.json")


class OriginalTensorReader:
    """Bounded, network-free original tensor spans for dense/head/embedding use.

    `read_into(name, byte_offset, target)` fills a caller-owned writable
    contiguous buffer. Each call is at most 8 MiB; no returned array or hidden
    model-wide mapping is created. Retained metadata is explicitly selected.
    """

    def __init__(self, metadata, originals, integrity, budget, *,
                 tensor_names=("head.weight", "embed.weight"), workspace=None):
        if (not isinstance(tensor_names, (list, tuple)) or not 1 <= len(tensor_names) <= 4096
                or any(not isinstance(name, str) or not name for name in tensor_names)
                or len(set(tensor_names)) != len(tensor_names)):
            raise ValueError("Select one to 4096 unique original tensor names")
        self.workspace = _scope(workspace, metadata, originals, integrity)
        self.metadata, self.originals, self.integrity = map(Path, (metadata, originals, integrity))
        self.budget, self.names = budget, tuple(tensor_names)
        self.tensors, self.blobs = {}, OrderedDict()
        self._opened = False
        self.retired_data_bytes = self.retired_index_bytes = self.logical_bytes = self.read_calls = 0

    def __enter__(self):
        if self._opened:
            raise RuntimeError("Original tensor reader is already open")
        self.budget.acquire(EXPERT_METADATA_BYTES, category="metadata"); self._opened = True
        try:
            self.workspace.check()
            _metadata_paths(self.metadata)
            _regular_below(self.integrity, self.integrity / "manifest.json")
            self.inventory, self.receipts, self.registration_sha256 = _registration(self.metadata, self.integrity)
            catalog, _ = load_catalog(self.metadata)
            self.tensors = {name: catalog[name] for name in self.names}
            for t in self.tensors.values():
                if (t["dtype"] not in DTYPE_BYTES or t["shard"] not in self.inventory
                        or type(t["shard_byte_offset"]) is not int or t["shard_byte_offset"] < 0
                        or type(t["bytes"]) is not int or t["bytes"] <= 0
                        or t["shard_byte_offset"] + t["bytes"] > self.inventory[t["shard"]]["bytes"]):
                    raise ValueError("Original tensor extent or dtype differs")
            return self
        except BaseException:
            self.__exit__(); raise

    def _close_blob(self, blob):
        self.retired_data_bytes += blob.data.aligned_bytes_requested
        self.retired_index_bytes += blob.index.aligned_bytes_requested
        blob.close()

    def _blob(self, name):
        if name in self.blobs:
            self.blobs.move_to_end(name)
        else:
            if len(self.blobs) == 2:
                _, old = self.blobs.popitem(last=False); self._close_blob(old)
            _regular_below(self.originals, self.originals / name)
            _regular_below(self.integrity, self.integrity / (name + ".sha256"))
            self.blobs[name] = VerifiedBlob(self.originals / name, self.integrity,
                self.receipts[name], self.inventory[name], self.budget)
        return self.blobs[name]

    def read_into(self, name, byte_offset, target):
        if not self._opened:
            raise RuntimeError("Original tensor reader is closed")
        view = memoryview(target)
        if view.readonly or not view.c_contiguous:
            raise ValueError("Expected a caller-owned writable contiguous tensor buffer")
        view = view.cast("B")
        t = self.tensors[name]
        item_bytes = DTYPE_BYTES[t["dtype"]]
        if (type(byte_offset) is not int or byte_offset < 0 or byte_offset % item_bytes
                or not 0 < view.nbytes <= CHUNK_BYTES or view.nbytes % item_bytes
                or byte_offset + view.nbytes > t["bytes"]):
            raise ValueError("Original tensor read exceeds its aligned element extent or 8 MiB bound")
        self.budget.acquire(0)
        source = self._blob(t["shard"])
        for offset in range(0, view.nbytes, 4096):
            self.budget.acquire(0)
            raw = source.read(t["shard_byte_offset"] + byte_offset + offset, min(4096, view.nbytes - offset))
            view[offset:offset + len(raw)] = raw
        self.logical_bytes += view.nbytes; self.read_calls += 1

    def stats(self):
        return {"selected_tensors": len(self.tensors), "open_shard_readers": len(self.blobs),
                "logical_bytes": self.logical_bytes, "logical_reads": self.read_calls,
                "physical_data_bytes_read": self.retired_data_bytes + sum(b.data.aligned_bytes_requested for b in self.blobs.values()),
                "integrity_index_bytes_read": self.retired_index_bytes + sum(b.index.aligned_bytes_requested for b in self.blobs.values()),
                "maximum_span_request_bytes": CHUNK_BYTES, "maximum_verified_block_slice_bytes": 4096,
                "model_wide_mapping": False, "network_bytes_in_forward": 0}

    def __exit__(self, *_):
        first_error = None
        for blob in self.blobs.values():
            try:
                self._close_blob(blob)
            except BaseException as error:
                first_error = first_error or error
        self.blobs.clear(); self.tensors.clear()
        if self._opened:
            self.budget.release(EXPERT_METADATA_BYTES, category="metadata"); self._opened = False
        if first_error:
            raise first_error


class OriginalShardExpertStore:
    """NativeStore-compatible offline source for all 384 experts of one layer.

    SourceFactory example: lambda _, budget: OriginalShardExpertStore(
        metadata, originals, integrity, budget, layer=layer, workspace=ws).
    The caller owns one expert-sized writable target. Extra payload storage is
    only VerifiedBlob's bounded direct-I/O scratch and two cached blocks.
    """

    def __init__(self, metadata, originals, integrity, budget, *, layer, workspace=None):
        validate_expert_identity(layer, 0)
        self.workspace = _scope(workspace, metadata, originals, integrity)
        self.metadata, self.originals, self.integrity = map(Path, (metadata, originals, integrity))
        self.budget, self.layer, self.size = budget, layer, EXPERT_BYTES
        self.timings, self.gpu_integrity, self.expected_chunks = Timings(), False, {}
        self.entries, self.blobs, self.manifest = {}, OrderedDict(), None
        self.read_calls = self.bytes_read = self.file_switches = 0
        self.retired_data_bytes = self.retired_index_bytes = 0
        self._opened = False

    def __enter__(self):
        if self._opened:
            raise RuntimeError("Original expert source is already open")
        self.budget.acquire(EXPERT_METADATA_BYTES, category="metadata"); self._opened = True
        try:
            self.workspace.check()
            _metadata_paths(self.metadata)
            _regular_below(self.integrity, self.integrity / "manifest.json")
            self.inventory, self.receipts, registration_sha = _registration(self.metadata, self.integrity)
            catalog, identity = load_catalog(self.metadata)
            layout = tensor_layout()
            for expert in range(384):
                tensors = []
                for short, shape in EXPERT_LAYOUT.items():
                    projection, suffix = short.split("_")
                    name = f"layers.{self.layer}.ffn.experts.{expert}.{projection}.{suffix}"
                    t = catalog[name]
                    if (t["dtype"], t["shape"], t["bytes"]) != (
                            "I8" if suffix == "weight" else "F8_E8M0", list(shape), layout[short]["bytes"]):
                        raise ValueError("Original expert tensor layout differs")
                    if t["shard"] not in self.inventory or t["shard_byte_offset"] < 0 or t["shard_byte_offset"] + t["bytes"] > self.inventory[t["shard"]]["bytes"]:
                        raise ValueError("Original expert tensor extent differs")
                    tensors.append({**t, "tensor": name, "name": short})
                self.entries[f"{self.layer}:{expert}"] = {"bytes": self.size, "sha256": None,
                    "provenance": {"kind": "complete_expert", "model": MODEL, "revision": REVISION,
                        "layer": self.layer, "expert": expert, "swiglu_limit": 10.0, "tensors": tensors}}
            self.manifest = {"kind": "registered_original_shard_experts", "model": MODEL, "revision": REVISION,
                "layer": self.layer, "layout": layout, "expert_bytes": self.size, "experts": 384,
                "catalog_sha256": identity["catalog_sha256"], "registration_sha256": registration_sha,
                "payload_bytes_repacked": 0, "network_in_forward": False}
            self.manifest_sha256 = digest(_json(self.manifest))
            return self
        except BaseException:
            self.__exit__(); raise

    def _blob(self, name):
        if name in self.blobs:
            self.blobs.move_to_end(name)
        else:
            # Qualified released layouts put a layer's experts in one shard.
            # Still bound the handles if an unexpected cross-shard layout is
            # presented; no model-wide source handle or payload cache exists.
            if len(self.blobs) == 2:
                _, old = self.blobs.popitem(last=False)
                self.retired_data_bytes += old.data.aligned_bytes_requested
                self.retired_index_bytes += old.index.aligned_bytes_requested
                old.close()
            _regular_below(self.originals, self.originals / name)
            _regular_below(self.integrity, self.integrity / (name + ".sha256"))
            self.blobs[name] = VerifiedBlob(self.originals / name, self.integrity,
                self.receipts[name], self.inventory[name], self.budget)
            self.file_switches += 1
        return self.blobs[name]

    def validate(self, key, data, *, tensors=False):
        entry = self.entries[key]
        if not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.shape != (self.size,) or not data.flags.c_contiguous:
            raise ValueError("Expected exactly one contiguous original expert")
        if entry["sha256"] is None or digest(memoryview(data)) != entry["sha256"]:
            raise ValueError("Original expert has no verified source read or its bytes changed")
        if tensors:
            for t in entry["provenance"]["tensors"]:
                region = self.manifest["layout"][t["name"]]
                if digest(memoryview(data)[region["offset"]:region["offset"] + region["bytes"]]) != t["sha256"]:
                    raise ValueError("Original expert tensor bytes changed")

    def read_into(self, key, target):
        if not self._opened:
            raise RuntimeError("Original expert source is closed")
        if not isinstance(target, np.ndarray) or target.dtype != np.uint8 or target.shape != (self.size,) or not target.flags.c_contiguous or not target.flags.writeable:
            raise ValueError("Expected one caller-owned writable original expert buffer")
        entry = self.entries[key]
        self.budget.acquire(0)
        full_hash, hashes = hashlib.sha256(), {}
        view = memoryview(target)
        with self.timings.section("storage.direct_read"):
            for t in entry["provenance"]["tensors"]:
                source, sha = self._blob(t["shard"]), hashlib.sha256()
                region = self.manifest["layout"][t["name"]]
                for offset in range(0, t["bytes"], 4096):
                    self.budget.acquire(0)
                    raw = source.read(t["shard_byte_offset"] + offset, min(4096, t["bytes"] - offset))
                    view[region["offset"] + offset:region["offset"] + offset + len(raw)] = raw
                    sha.update(raw); full_hash.update(raw)
                hashes[t["name"]] = sha.hexdigest()
        actual = full_hash.hexdigest()
        if entry["sha256"] is not None and entry["sha256"] != actual:
            raise ValueError("Previously verified original expert identity changed")
        entry["sha256"] = actual
        for t in entry["provenance"]["tensors"]:
            t["sha256"] = hashes[t["name"]]
        self.validate(key, target, tensors=True)
        if self.gpu_integrity and key not in self.expected_chunks:
            shape = digest_shape(self.size); size = int(np.prod(shape))
            self.budget.acquire(size, category="integrity")
            try:
                result = np.empty(shape, np.uint8)
                hash_chunks(target, result); result.setflags(write=False)
                self.expected_chunks[key] = result
            except BaseException:
                self.budget.release(size, category="integrity"); raise
        self.read_calls += 1; self.bytes_read += self.size

    def stats(self):
        return {"experts_available": len(self.entries), "logical_expert_reads": self.read_calls,
                "logical_expert_bytes": self.bytes_read, "open_shard_readers": len(self.blobs),
                "physical_data_bytes_read": self.retired_data_bytes + sum(b.data.aligned_bytes_requested for b in self.blobs.values()),
                "integrity_index_bytes_read": self.retired_index_bytes + sum(b.index.aligned_bytes_requested for b in self.blobs.values()),
                "payload_bytes_repacked": 0, "network_payload_bytes_in_forward": 0,
                "maximum_verified_span_read_bytes": 4096}

    def __exit__(self, *_):
        first_error = None
        for blob in self.blobs.values():
            try:
                self.retired_data_bytes += blob.data.aligned_bytes_requested
                self.retired_index_bytes += blob.index.aligned_bytes_requested
                blob.close()
            except BaseException as error:
                first_error = first_error or error
        self.blobs.clear()
        for value in self.expected_chunks.values():
            self.budget.release(value.nbytes, category="integrity")
        self.expected_chunks.clear()
        self.entries.clear(); self.manifest = None
        if self._opened:
            self.budget.release(EXPERT_METADATA_BYTES, category="metadata"); self._opened = False
        if first_error:
            raise first_error
