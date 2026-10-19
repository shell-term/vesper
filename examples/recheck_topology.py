"""Replay a complete Attention-sharing group with NumPy and native CUDA."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_topology import role
from vesper.compressed_attention_validation import profiles, cleanup
from vesper.compression import NativeCompression
from vesper.indexer import NativeIndexer
from vesper.normalization import NativeNormRoPE
from vesper.projections import NativeProjections
from vesper.resources import MemoryBudget, MiB
from vesper.topology_comparison import load_group
from vesper.topology_validation import validate_goldens, exercise, replay_comparison, fixture_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-report", "fixture-directory", "attention-bundle", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 96 * MiB)
    reference = json.loads(raw)
    goldens = validate_goldens(reference)
    layers = tuple(reference["layers"])
    budget = MemoryBudget(cap=2048 * MiB)
    budget.acquire(2048 * MiB)
    try:
        arrays, fixtures = load_group(args.fixture_directory, layers)
        if fixture_identities(fixtures) != fixture_identities(reference["fixtures"]): raise ValueError("Original sharing fixture identities differ")
        seed_goldens = {s["id"]: s["meta"] for s in reference["diagnostic_seeds"]}
        with ExitStack() as stack:
            ps = {l: stack.enter_context(NativeProjections(arrays, layer=l)) for l in layers}
            norm = stack.enter_context(NativeNormRoPE())
            comp = stack.enter_context(NativeCompression(arrays, layers=tuple(l for l in layers if role(l).owns_kv)))
            index = stack.enter_context(NativeIndexer(arrays, layers=tuple(l for l in layers if role(l).owns_index)))
            attn = stack.enter_context(NativeAttention(args.attention_bundle, profile="long_context"))
            ops = ps, norm, comp, index, attn
            native_profiles = profiles(ops)
            if native_profiles != reference["native_profiles"]: raise ValueError("Sharing native profiles differ")
            def checker(m, xs, actual):
                golden = goldens[m["id"]]
                if m != golden["meta"]: raise ValueError("Sharing replay input metadata differs")
                return replay_comparison(golden, actual, reference["boundary_names"])
            def seed_reference(identity, state, m):
                if m != seed_goldens[identity]: raise ValueError("Original/native sharing seed identity differs")
            rows, faults, events, seeds, minimum = exercise(layers, arrays, ops, checker, lambda *_: None,
                seed_reference, lambda *_: None, lambda p: print(json.dumps(p), flush=True))
        cleaned = cleanup(ops)
        frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
        frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
        passed = (all(r["status"] == "pass" for r in rows + faults + events) and not any(frameworks[n + "_imported"] for n in ("torch", "tilelang"))
            and all(c["device_owned_bytes"] == 0 and c["host_reserved_bytes"] == 0 and not c["owned_context_active"] for c in cleaned.values()))
        result = {"schema_version": 1, "probe": "native_attention_topology_recheck", "status": "pass" if passed else "fail", "layers": list(layers),
            "reference_report_sha256": hashlib.sha256(raw).hexdigest(), **frameworks, "native_profiles": native_profiles,
            "frame_cases": rows, "faults": faults, "snapshot_events": events, "diagnostic_seeds": seeds,
            "memory": {"cleanup": cleaned, "combined_native_host_allowance_bytes": (512 * len(layers) + 3872) * MiB,
                "retained_fixture_host_allowance_bytes": 2048 * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": True, "state_storage": "bounded_host_diagnostic_frame", "reservation_is_RSS_limit": False},
            "weights_fetched_during_recheck_bytes": 0, "full_model_quality": "not_run", "end_to_end_speed": "not_measured", "scope": reference["scope"]}
    finally: budget.release(2048 * MiB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "layers": list(layers), "frames": len(rows), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
