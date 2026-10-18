"""Replay original compressor/FP4-cache goldens without Torch or TileLang."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.attention_validation import array_digest
from vesper.compression import NativeCompression
from vesper.compression_fixture import load_compression_fixture
from vesper.compression_validation import exercise_components, exercise_states, exercise_failures, exercise_snapshots, validate_goldens
from vesper.resources import MemoryBudget, MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-report", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 32 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    budget = MemoryBudget(cap=128 * MiB)
    budget.acquire(128 * MiB)
    try:
        arrays, fixture = load_compression_fixture(args.fixture)
    finally:
        budget.release(128 * MiB)
    if fixture != report["fixture"]:
        raise ValueError("Original compressor fixture provenance differs")
    def checker(meta, values, actual):
        golden = goldens[meta["id"]]
        if meta != golden["meta"] or set(actual) != set(golden["stage_sha256"]):
            raise ValueError("Compression input or boundary set differs")
        return [{"check": n, "status": "pass" if array_digest(a) == golden["stage_sha256"][n] else "fail", "elements": int(a.size),
                 "expected_sha256": golden["stage_sha256"][n], "actual_sha256": array_digest(a)} for n, a in actual.items()]
    with NativeCompression(arrays) as native:
        profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256, "device": native.info,
                   "cublas_version": native.blas.version, "cublas_math_mode": native.blas.math_mode}
        norm_profile = {"compiler": native.norm.compiler, "cubin_sha256": native.norm.cubin_sha256}
        if profile != report["compression_profile"] or norm_profile != report["normalization_profile"]:
            raise ValueError("Registered compression or normalization profile differs")
        components, minimum = exercise_components(native, checker)
        states, low = exercise_states(native, checker, lambda *_: None)
        retries, faults = exercise_failures(native, checker, lambda *_: None)
        snapshots, events = exercise_snapshots(native, checker, lambda *_: None, lambda *_: None)
        states.extend(retries + snapshots)
        minimum = min(minimum, low)
    frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
    frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
    passed = (all(r["status"] == "pass" for r in components + states + faults + events)
              and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
              and not native.stats()["device_owned_bytes"] and not native.norm.stats()["device_owned_bytes"])
    result = {"schema_version": 1, "probe": "native_compression_recheck", "status": "pass" if passed else "fail",
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), **frameworks,
              "compression_profile": profile, "normalization_profile": norm_profile, "component_cases": components,
              "state_cases": states, "faults": faults, "snapshot_events": events,
              "memory": {"compression_cleanup": native.stats(), "normalization_cleanup": native.norm.stats(),
                  "combined_native_host_allowance_bytes": 240 * MiB, "all_state_reservations_released": True,
                  "minimum_sampled_host_available_bytes": minimum, "publication_cache": "bounded_host_diagnostic_copy", "reservation_is_RSS_limit": False},
              "weights_fetched_bytes": 0, "index_selection": "not_qualified", "source_layer_sharing": "not_implemented",
              "complete_compressed_attention": "not_implemented", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "components": len(components), "state_calls": len(states), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
