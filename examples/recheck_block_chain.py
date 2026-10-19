"""Reproduce every complete 20..25 chain golden with NumPy/native CUDA only."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.block_chain_validation import load_arrays, open_operators, open_chain, validate_goldens, exercise, replay_comparison, state_digest
from vesper.block_chain_validation import FIXTURE_HOST_MIB, NATIVE_HOST_MIB
from vesper.compressed_attention_validation import profiles, cleanup
from vesper.compressed_block_validation import fixture_identity
from vesper.resources import MemoryBudget, MiB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("reference-report", "topology-fixtures", "block-root", "attention-bundle", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    raw = read_bounded(a.reference_report, 64 * MiB)
    reference = json.loads(raw); goldens = validate_goldens(reference)
    budget = MemoryBudget(cap=FIXTURE_HOST_MIB * MiB); budget.acquire(budget.cap)
    try:
        arrays, fixtures = load_arrays(a.topology_fixtures, a.block_root)
        if fixture_identity(fixtures) != fixture_identity(reference["fixtures"]): raise ValueError("Chain original fixtures differ")
        with ExitStack() as stack:
            hc, ops = open_operators(stack, arrays, a.attention_bundle)
            chain = open_chain(stack, arrays, hc, ops, a.block_root)
            native_profiles = profiles(ops)
            hc_profiles = {str(l): {"compiler": h.compiler, "cubin_sha256": h.cubin_sha256, "cublas_version": h.blas.version, "cublas_math_mode": h.blas.math_mode} for l, h in hc.items()}
            if native_profiles != reference["native_profiles"] or hc_profiles != reference["hc_profiles"]: raise ValueError("Chain native binary/compiler identities differ")
            for l, m in chain.moe.items():
                expected = reference["moe_fixtures"][str(l)]
                if (any(v != expected[k] for k, v in (("gate", m.gate_manifest), ("router", m.router_manifest), ("shared", m.shared_manifest)))
                        or sorted((e["provenance"] for e in m.experts.store.entries.values()), key=lambda e: e["expert"]) != sorted(expected["experts"], key=lambda e: e["expert"])):
                    raise ValueError("Chain original MoE artifacts differ")
            allowance = (budget.cap + chain.budget.cap + chain.attention.budget.cap + sum(h.budget.cap for h in hc.values())
                + sum(p.budget.cap for p in ops[0].values()) + sum(o.budget.cap for o in (*ops[1:], ops[2].norm, ops[3].norm))
                + sum(m.budget.cap for m in chain.moe.values()))
            if allowance != NATIVE_HOST_MIB * MiB: raise ValueError("Chain native allowance omits a ledger")
            def checker(m, h, p, actual, state):
                golden = goldens[m["id"]]
                if m != golden["meta"] or state_digest(state) != golden["committed_state_sha256"]: raise ValueError("Native chain entry or committed state differs")
                return replay_comparison(golden, actual, reference["boundary_names"])
            rows, faults, events, minimum = exercise(chain, ops, reference["plan"], checker, lambda: None, lambda *x: None,
                lambda x: print(json.dumps(x), flush=True))
            chain_stats, moe_stats = chain.stats(), {str(l): m.snapshot() for l, m in chain.moe.items()}
        cleaned = cleanup(ops)
        frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
        frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
        passed = (all(r["status"] == "pass" for r in rows + faults + events) and not any(frameworks[n + "_imported"] for n in ("torch", "tilelang"))
            and all(not c["device_owned_bytes"] and not c["host_reserved_bytes"] and not c["owned_context_active"] for c in cleaned.values())
            and all(not h.stats()["device_owned_bytes"] and not h.stats()["host_reserved_bytes"] and not h.stats()["owned_context_active"] for h in hc.values())
            and all(not any(m.cleanup.values()) for m in chain.moe.values()) and not chain.budget.reserved and not chain.attention.budget.reserved)
        r = {"schema_version": 1, "probe": "complete_refresh_chain_recheck", "status": "pass" if passed else "fail", "layers": reference["layers"],
            "reference_report_sha256": hashlib.sha256(raw).hexdigest(), **frameworks, "native_profiles": native_profiles, "hc_profiles": hc_profiles,
            "chain_cases": rows, "faults": faults, "snapshot_events": events,
            "memory": {"operator_cleanup": cleaned, "HC_cleanup": {str(l): h.stats() for l, h in hc.items()}, "moe_cleanup": {str(l): m.cleanup for l, m in chain.moe.items()},
                "chain_cleanup": chain.stats(), "chain_active_stats": chain_stats, "moe_active_stats": moe_stats,
                "combined_native_host_allowance_bytes": allowance, "retained_fixture_host_allowance_bytes": FIXTURE_HOST_MIB * MiB,
                "minimum_sampled_host_available_bytes": minimum, "all_frame_reservations_released": chain.attention.budget.reserved == 0,
                "reservation_is_RSS_limit": False, "state_storage": "bounded_host_diagnostic_frame"},
            "weights_fetched_during_recheck_bytes": 0, "scope": reference["scope"], "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally: budget.release(budget.cap)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes((json.dumps(r, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": r["status"], "chain_steps": len(rows), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__": raise SystemExit(main())
