"""Exact outer KV checkpoint recovery over real ring wraps and failed I/O."""

from unittest.mock import patch
import numpy as np

from .attention_validation import array_digest
from .kv import NativeWindowKV
from .kv_validation import kv_inputs


def probe():
    cases = []
    for batch in (1, 2):
        kv = NativeWindowKV(batch)
        with kv:
            kv.append(kv_inputs(batch, 128, batch, "finite_bf16_encodings"))
            before = kv.inspect()
            handle = kv.checkpoint()
            inner = kv.snapshot()
            kv.append(kv_inputs(batch, 1, 10 + batch))
            kv.discard(inner)
            kv.append(kv_inputs(batch, 1, 20 + batch))
            advanced = kv.inspect()
            with patch.object(kv, "_cache", side_effect=OSError("Injected checkpoint readback failure")):
                try:
                    kv.restore(handle)
                except OSError:
                    pass
                else:
                    raise RuntimeError("Failed checkpoint readback was published")
            failed = kv.inspect()
            if failed["position"] != advanced["position"] or not np.array_equal(failed["cache_bf16"], advanced["cache_bf16"]):
                raise RuntimeError("Failed outer restore changed active ring state")
            restored = kv.restore(handle)
            if restored["position"] != before["position"] or not np.array_equal(restored["cache_bf16"], before["cache_bf16"]):
                raise RuntimeError("Outer KV restore changed ring storage bits")
            retried = kv.append(kv_inputs(batch, 1, 10 + batch))
            expected = advanced["cache_bf16"].copy()
            # The second append overwrote slot one only; after rollback and
            # replay, slot zero is the first rounded token and slot one is old.
            expected[:, 1] = before["cache_bf16"][:, 1]
            if not np.array_equal(retried["cache_bf16"], expected):
                raise RuntimeError("Outer KV retry changed rounded ring bytes")
            active = kv.stats()
        cleanup = kv.stats()
        if cleanup["device_owned_bytes"] or cleanup["host_reserved_bytes"] or cleanup["outer_host_checkpoint_bytes"]:
            raise RuntimeError("Outer KV probe retained owned buffers")
        cases.append({"batch": batch, "status": "pass", "checkpoint_position": 128, "advanced_position": 130,
            "restored_position": restored["position"], "original_ring_sha256": array_digest(before["cache_bf16"]),
            "restored_ring_sha256": array_digest(restored["cache_bf16"]), "retry_ring_sha256": array_digest(retried["cache_bf16"]),
            "readback_failure_kept_active_state": True, "host_checkpoint_bytes": active["outer_host_checkpoint_peak_bytes"],
            "active": active, "cleanup": cleanup})
    return {"probe": "native_outer_KV_ring_checkpoint", "status": "pass", "cases": cases,
        "scope": "State-copy/recovery probe; original rounding arithmetic was independently qualified earlier"}
