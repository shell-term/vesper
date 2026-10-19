"""Replay full compressed-Attention goldens with NumPy and registered CUDA."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_validation import array_digest
from vesper.compressed_attention_fixture import load_compressed_attention_fixture
from vesper.compressed_attention_validation import validate_goldens, exercise, profiles, cleanup
from vesper.compression import NativeCompression
from vesper.compression_fixture import load_compression_fixture
from vesper.indexer import NativeIndexer
from vesper.indexer_fixture import load_indexer_fixture
from vesper.normalization import NativeNormRoPE
from vesper.projections import NativeProjections
from vesper.resources import MemoryBudget, MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-report", "fixture", "compression-fixture", "indexer-fixture", "attention-bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 64 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    budget = MemoryBudget(cap=768 * MiB)
    budget.acquire(768 * MiB)
    try:
        arrays, fixture = load_compressed_attention_fixture(args.fixture)
        comp_arrays, comp_fixture = load_compression_fixture(args.compression_fixture)
        idx_arrays, idx_fixture = load_indexer_fixture(args.indexer_fixture)
    finally: budget.release(768 * MiB)
    if {"attention": fixture, "compression": comp_fixture, "indexer": idx_fixture} != report["fixtures"]:
        raise ValueError("Original connected Attention fixture identities differ")
    with ExitStack() as stack:
        ps = {l: stack.enter_context(NativeProjections(arrays, layer=l)) for l in (2, 20, 24)}
        norm = stack.enter_context(NativeNormRoPE())
        comp = stack.enter_context(NativeCompression(comp_arrays))
        index = stack.enter_context(NativeIndexer(idx_arrays))
        attn = stack.enter_context(NativeAttention(args.attention_bundle))
        ops = ps, norm, comp, index, attn
        profile = profiles(ops)
        if profile != report["native_profiles"]: raise ValueError("Registered full Attention profiles differ")
        def checker(meta, xs, actual):
            golden = goldens[meta["id"]]
            if meta != golden["meta"] or set(actual) != set(golden["stage_sha256"]): raise ValueError("Complete Attention replay input/boundaries differ")
            return [{"check": n, "status": "pass" if array_digest(a) == golden["stage_sha256"][n] else "fail", "elements": int(a.size),
                "expected_sha256": golden["stage_sha256"][n], "actual_sha256": array_digest(a)} for n, a in actual.items()]
        rows, faults, events, minimum = exercise(arrays, ops, checker, lambda *_: None, lambda *_: None)
    cleaned = cleanup(ops)
    frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
    frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
    passed = (all(r["status"] == "pass" for r in rows + faults + events) and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
              and all(not c["device_owned_bytes"] and not c["host_reserved_bytes"] and not c["owned_context_active"] for c in cleaned.values()))
    result = {"schema_version": 1, "probe": "native_compressed_attention_recheck", "status": "pass" if passed else "fail",
        "reference_report_sha256": hashlib.sha256(raw).hexdigest(), **frameworks, "native_profiles": profile,
        "frame_cases": rows, "faults": faults, "snapshot_events": events,
        "memory": {"cleanup": cleaned, "combined_native_host_allowance_bytes": 2624 * MiB, "minimum_sampled_host_available_bytes": minimum,
            "all_frame_reservations_released": True, "state_storage": "bounded_host_diagnostic_frame", "reservation_is_RSS_limit": False},
        "weights_fetched_during_recheck_bytes": 0, "scope": report["scope"], "other_layers_and_larger_context": "not_qualified",
        "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "frames": len(rows), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
