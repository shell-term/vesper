"""Offline original-route MoE using a payload-sharing expert index."""

from .backbone_prepare import PreparedMoE
from .backbone_runtime import validate_moe_manifests
from .connected import ConnectedMoE
from .indexed_store import IndexedStore


class IndexedMoE(PreparedMoE):
    def __init__(self, root, bindings, layer, prepared, expected_provenance, *, index_directory):
        super().__init__(root, bindings, layer, prepared, expected_provenance)
        self.store = index_directory
        self.calls, self.reads, self.bytes_read, self.file_switches = 0, 0, 0, 0
        self.index_sha256 = None

    def forward(self, values, image_span=False):
        factory = lambda directory, budget: IndexedStore(directory, self.root, budget, layer=self.layer)
        with ConnectedMoE(self.store, *self.files, layer=self.layer, profile="backbone_probe", source_factory=factory) as moe:
            validate_moe_manifests(moe, self.bindings, self.layer)
            source = moe.experts.store
            if set(source.entries) != {f"{self.layer}:{e}" for e in self.available} or any(
                    e["provenance"] != self.expected_provenance[k] for k, e in source.entries.items()):
                raise ValueError("Indexed original expert provenance differs from the independent capture")
            if self.index_sha256 is not None and self.index_sha256 != source.manifest_sha256:
                raise ValueError("Immutable expert index changed between model calls")
            self.index_sha256 = source.manifest_sha256
            result = moe.forward(values, image_span)
            self.calls += 1; self.reads += source.read_calls; self.bytes_read += source.bytes_read; self.file_switches += source.file_switches
            return result

    def stats(self):
        return {"layer": self.layer, "calls": self.calls, "original_experts": sorted(self.available),
            "index_sha256": self.index_sha256, "direct_read_calls": self.reads, "direct_read_bytes": self.bytes_read,
            "payload_file_opens_and_switches": self.file_switches, "payload_bytes_copied_to_build_index": 0,
            "expert_payloads_fetched_in_forward": 0}
