"""Read-only indexes over immutable packed stores without copying their payloads.

One index contains at most one layer's 384 original experts. A shared declared
source root anchors every relative path; escapes and changed source manifests
fail before payload I/O. There is one aligned scratch buffer and at most one
retained file handle, regardless of the number of indexed experts/files.
"""

import json
from pathlib import Path
import re

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, validate_expert_identity
from .model_artifacts import canonical, digest
from .resources import MemoryBudget, MiB
from .storage import NativeStore, tensor_layout
from .disk import DirectReader
from .timing import Timings

MAX_INDEX_EXPERTS, MAX_INDEX_BYTES, INDEX_METADATA_BYTES = 384, 4 * MiB, 32 * MiB
KIND = "vesper_indexed_native_experts"


def _metadata(path, limit):
    if not 1 <= path.stat().st_size <= limit: raise ValueError("Excessive or empty expert index/source metadata")
    with path.open("rb") as f: raw = f.read(limit + 1)
    if len(raw) > limit: raise ValueError("Expert metadata grew beyond its bound")
    return json.loads(raw), raw


def _relative(root, directory):
    root, directory = Path(root).resolve(strict=True), Path(directory).resolve(strict=True)
    if not directory.is_relative_to(root): raise ValueError("Expert source escapes the declared source root")
    name = directory.relative_to(root).as_posix()
    if name == "." or len(name) > 512: raise ValueError("Expected a bounded relative expert-store directory")
    return name


def _source(root, name, budget):
    if type(name) is not str: raise ValueError("Invalid anchored original-store path")
    path = Path(name)
    if (len(name) > 512 or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or "\\" in name or re.match(r"^[A-Za-z]:", name)):
        raise ValueError("Invalid anchored original-store path")
    directory = (root / path).resolve(strict=True)
    if _relative(root, directory) != name: raise ValueError("Original source path resolves outside its declared location")
    m, raw = _metadata(directory / "manifest.json", MiB)
    # The existing parser retains every format/provenance/extent check. Bind
    # the parsed object to the exact bytes we hashed, including during races.
    packed = NativeStore(directory, budget, max_experts=48)
    if packed.manifest != m: raise ValueError("Original source manifest changed during validation")
    file = directory / "experts.bin"
    if not file.resolve(strict=True).is_relative_to(root): raise ValueError("Expert payload escapes the declared source root")
    return packed, digest(raw)


def build_index(sources, directory, source_root, *, layer, experts=None):
    """Index existing qualified-format stores; zero payload reads or writes.

    An optional expert list restricts a test fixture's available IDs, never the
    router's choices. A requested unavailable original expert is an error.
    Duplicate IDs must have identical payload and provenance; the first source
    in deterministic path order wins. Existing indexes are never replaced.
    """
    validate_expert_identity(layer, 0)
    if type(sources) not in (list, tuple) or not 1 <= len(sources) <= MAX_INDEX_EXPERTS:
        raise ValueError("Expected one to 384 bounded packed sources")
    selected = None if experts is None else list(experts)
    if selected is not None and (not 1 <= len(selected) <= MAX_INDEX_EXPERTS or selected != sorted(set(selected))
            or any(type(e) is not int or not 0 <= e < 384 for e in selected)):
        raise ValueError("Invalid original expert selection for the index")
    directory, root = Path(directory).resolve(), Path(source_root).resolve(strict=True)
    if directory.exists(): raise FileExistsError("Refusing to replace an expert index")
    names = sorted({_relative(root, p) for p in sources})
    budget = MemoryBudget(cap=INDEX_METADATA_BYTES); budget.acquire(budget.cap, category="metadata")
    try:
        entries, receipts = {}, []
        for name in names:
            packed, sha = _source(root, name, budget)
            ordinal = len(receipts)
            receipts.append({"directory": name, "manifest_sha256": sha, "file_bytes": packed.manifest["file_bytes"]})
            for key, entry in packed.entries.items():
                if entry["provenance"]["layer"] != layer: raise ValueError("An indexed source belongs to a different layer")
                if selected is not None and entry["provenance"]["expert"] not in selected: continue
                if key in entries:
                    if any(entries[key][n] != entry[n] for n in ("bytes", "sha256", "provenance")):
                        raise ValueError("Conflicting original bytes/provenance for a duplicate expert")
                else: entries[key] = {**entry, "source": ordinal}
            del packed
        ids = sorted(e["provenance"]["expert"] for e in entries.values())
        if not ids or len(ids) > 384 or (selected is not None and ids != selected):
            raise ValueError("Missing requested original experts or excessive index size")
        manifest = {"schema_version": 1, "kind": KIND, "model": MODEL, "revision": REVISION, "layer": layer,
            "maximum_experts": MAX_INDEX_EXPERTS, "expert_bytes": EXPERT_BYTES, "layout": tensor_layout(),
            "sources": receipts, "entries": sorted(entries.values(), key=lambda e: e["provenance"]["expert"]),
            "payload_bytes_copied": 0, "weight_payloads_fetched": 0}
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        if len(raw) > MAX_INDEX_BYTES: raise ValueError("Expert index metadata exceeds its explicit bound")
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "manifest.json").open("xb") as f: f.write(raw)
        return {"probe": "build_indexed_native_store", "status": "pass", "experts": len(entries), "sources": len(receipts),
            "manifest_sha256": digest(raw), "semantic_sha256": canonical(manifest), "payload_bytes_copied": 0,
            "weight_payloads_fetched": 0, "indexed_original_bytes": len(entries) * EXPERT_BYTES}
    finally: budget.release(budget.cap, category="metadata")


class IndexedStore(NativeStore):
    """The same integrity-checked expert source interface, with lazy file switches."""

    def __init__(self, directory, source_root, budget, *, layer):
        validate_expert_identity(layer, 0)
        self.directory, self.source_root, self.layer = Path(directory).resolve(strict=True), Path(source_root).resolve(strict=True), layer
        self.budget, self.reader, self.current_source = budget, None, None
        self.timings, self.gpu_integrity, self.expected_chunks = Timings(), False, {}
        self.entries, self.manifest, self.paths = {}, None, {}
        self.size = EXPERT_BYTES
        self.read_calls = self.bytes_read = self.file_switches = 0
        self._metadata_owned = False

    def __enter__(self):
        if self._metadata_owned: raise RuntimeError("Indexed store is already open")
        self.budget.acquire(INDEX_METADATA_BYTES, category="metadata"); self._metadata_owned = True
        try:
            m, raw = _metadata(self.directory / "manifest.json", MAX_INDEX_BYTES)
            if (m.get("schema_version"), m.get("kind"), m.get("model"), m.get("revision"), m.get("layer"),
                    m.get("maximum_experts"), m.get("expert_bytes"), m.get("layout"), m.get("payload_bytes_copied"), m.get("weight_payloads_fetched")) != (
                    1, KIND, MODEL, REVISION, self.layer, 384, EXPERT_BYTES, tensor_layout(), 0, 0):
                raise ValueError("Indexed original expert identity/layout differs")
            receipts, entries = m.get("sources", []), m.get("entries", [])
            if not 1 <= len(receipts) <= 384 or not 1 <= len(entries) <= 384: raise ValueError("Excessive indexed sources/experts")
            names = [r["directory"] for r in receipts]
            if names != sorted(set(names)): raise ValueError("Unordered or duplicate indexed source paths")
            for ordinal, receipt in enumerate(receipts):
                packed, sha = _source(self.source_root, receipt["directory"], self.budget)
                if (sha, packed.manifest["file_bytes"]) != (receipt["manifest_sha256"], receipt["file_bytes"]):
                    raise ValueError("Indexed source manifest/extent changed")
                self.paths[ordinal] = packed.path
                expected = {e["key"]: e for e in entries if e.get("source") == ordinal}
                if any(key not in packed.entries or {k: v for k, v in e.items() if k != "source"} != packed.entries[key]
                        for key, e in expected.items()): raise ValueError("Indexed expert differs from its original packed source")
                del packed
            ids = []
            for e in entries:
                expert = e["provenance"]["expert"]; validate_expert_identity(self.layer, expert)
                if e["provenance"]["layer"] != self.layer or e["key"] != f"{self.layer}:{expert}": raise ValueError("Cross-layer indexed expert")
                if type(e.get("source")) is not int or e["source"] not in self.paths: raise ValueError("Invalid expert source ordinal")
                ids.append(expert); self.entries[e["key"]] = e
            if ids != sorted(set(ids)): raise ValueError("Duplicate or unordered indexed original experts")
            self.manifest, self.manifest_sha256 = m, digest(raw)
            # No payload handle/buffer exists until the first selected expert.
            return self
        except BaseException:
            self.__exit__(); raise

    def read_into(self, key, target):
        if not self._metadata_owned: raise RuntimeError("Indexed store is closed")
        entry = self.entries[key]; source = entry["source"]
        if self.reader is None:
            self.reader = DirectReader(self.paths[source], self.size, self.budget)
            self.current_source = source; self.file_switches += 1
        elif self.current_source != source:
            self.reader.switch_file(self.paths[source])
            self.current_source = source; self.file_switches += 1
        return super().read_into(key, target)

    def __exit__(self, *_):
        try: super().__exit__()
        finally:
            self.current_source = None; self.paths = {}; self.entries = {}; self.manifest = None
            if self._metadata_owned:
                self.budget.release(INDEX_METADATA_BYTES, category="metadata"); self._metadata_owned = False
