"""Replay every complete layer-2 block boundary without Torch or TileLang."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.compressed_attention_validation import profiles, cleanup
from vesper.compressed_block_validation import load_arrays, open_operators, open_block, validate_goldens, exercise, replay_comparison, fixture_identity
from vesper.resources import MemoryBudget, MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-report", "topology-fixtures", "fixtures", "attention-bundle", "store", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 32 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    budget = MemoryBudget(cap=768 * MiB)
    budget.acquire(768 * MiB)
    try:
        arrays, fixtures = load_arrays(args.topology_fixtures, args.fixtures)
        if fixture_identity(fixtures) != fixture_identity(report["fixtures"]): raise ValueError("Original layer-2 fixture identities differ")
        with ExitStack() as stack:
            hc, ops = open_operators(stack, arrays, args.attention_bundle)
            block = open_block(stack, arrays, hc, ops, args.store, args.fixtures)
            native_profiles = profiles(ops)
            hc_profile = {"compiler": hc.compiler, "cubin_sha256": hc.cubin_sha256, "cublas_version": hc.blas.version, "cublas_math_mode": hc.blas.math_mode}
            moe_fixtures = {"gate": block.moe.gate_manifest, "router": block.moe.router_manifest, "shared": block.moe.shared_manifest,
                "experts": [e["provenance"] for e in block.moe.experts.store.entries.values()]}
            expected_moe = report["moe_fixtures"]
            if (native_profiles != report["native_profiles"] or hc_profile != report["hc_profile"]
                    or any(moe_fixtures[n] != expected_moe[n] for n in ("gate", "router", "shared"))
                    or sorted(moe_fixtures["experts"], key=lambda e: e["expert"]) != sorted(expected_moe["experts"], key=lambda e: e["expert"])):
                raise ValueError("Complete block binary or original MoE fixture identities differ")
            def checker(m, hidden, pre, actual, state):
                from vesper.topology_validation import state_digest
                golden = goldens[m["id"]]
                if m != golden["meta"] or state_digest(state) != golden["committed_state_sha256"]:
                    raise ValueError("Complete block replay input/state metadata differs")
                return replay_comparison(golden, actual, report["boundary_names"])
            rows, faults, events, minimum = exercise(block, ops, report["plan"], checker, lambda: None, lambda *_: None,
                lambda p: print(json.dumps(p), flush=True))
            block_stats, moe_stats = block.stats(), block.moe.snapshot()
        cleaned = cleanup(ops)
        frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
        frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
        passed = (all(r["status"] == "pass" for r in rows + faults + events) and not any(frameworks[n + "_imported"] for n in ("torch", "tilelang"))
            and all(not c["device_owned_bytes"] and not c["host_reserved_bytes"] and not c["owned_context_active"] for c in cleaned.values())
            and not hc.stats()["device_owned_bytes"] and not hc.stats()["host_reserved_bytes"] and not any(block.moe.cleanup.values()) and not block.budget.reserved)
        result = {"schema_version": 1, "probe": "native_compressed_block_recheck", "status": "pass" if passed else "fail", "layer": 2,
            "reference_report_sha256": hashlib.sha256(raw).hexdigest(), **frameworks, "native_profiles": native_profiles, "hc_profile": hc_profile,
            "block_cases": rows, "faults": faults, "snapshot_events": events,
            "memory": {"operator_cleanup": cleaned, "HC_cleanup": hc.stats(), "moe_cleanup": block.moe.cleanup, "block_cleanup": block.stats(),
                "block_active_stats": block_stats, "moe_active_stats": moe_stats, "combined_native_host_allowance_bytes": 3584 * MiB,
                "retained_fixture_host_allowance_bytes": 768 * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": block.attention.budget.reserved == 0, "reservation_is_RSS_limit": False,
                "state_storage": "bounded_host_diagnostic_frame"},
            "weights_fetched_during_recheck_bytes": 0, "full_model_quality": "not_run", "end_to_end_speed": "not_measured", "scope": report["scope"]}
    finally: budget.release(768 * MiB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "blocks": len(rows), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
