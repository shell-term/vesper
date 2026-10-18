"""Recheck published shared/aggregation goldens without importing PyTorch."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from vesper.checkpoint import load_expert_fixture
from vesper.expert import ResidentExpert
from vesper.formats import bf16_bits
from vesper.moe import HIDDEN, MoEAccumulator
from vesper.moe_reference import aggregation_inputs, original_expert_inputs
from vesper.resources import MemoryBudget, MiB
from vesper.shared import ResidentSharedExpert, load_shared_fixture
from vesper.shared_reference import digest, expert_stages, shared_inputs


def read_report(path, probe):
    if path.stat().st_size > 8 * MiB:
        raise ValueError("Reference report exceeds this bounded example's size")
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("probe") != probe or report.get("status") != "pass":
        raise ValueError("Expected a passing report from the named original-reference probe")
    return report, hashlib.sha256(raw).hexdigest()


def check(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Published reference boundary set differs from the candidate")
    return {name: "pass" if digest(array) == expected[name] else "fail" for name, array in actual.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-fixture", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, nargs=6, required=True)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    shared_report, shash = read_report(args.reports_directory / "shared-expert.json", "shared_expert_parity")
    aggregation_report, ahash = read_report(args.reports_directory / "moe-aggregation.json", "moe_aggregation_parity")
    original_report, ohash = read_report(args.reports_directory / "original-expert-accumulation.json", "original_expert_accumulation")
    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    devices, cases, cubins, peaks = [], [], {}, {}
    try:
        shared_arrays, shared_manifest = load_shared_fixture(args.shared_fixture)
        loaded = [load_expert_fixture(path) for path in args.fixtures]
        loaded.sort(key=lambda pair: (pair[1]["layer"], pair[1]["expert"]))
        if (shared_report["fixture"] != shared_manifest or original_report["shared_fixture"] != shared_manifest
                or original_report["fixtures"] != [manifest for _, manifest in loaded]):
            raise ValueError("Original reference and candidate fixture manifests differ")
        with ExitStack() as stack:
            shared = stack.enter_context(ResidentSharedExpert(shared_arrays, args.device))
            experts = [stack.enter_context(ResidentExpert(arrays, args.device)) for arrays, _ in loaded]
            accumulator = stack.enter_context(MoEAccumulator(args.device))
            devices = [shared.device, *(expert.device for expert in experts), accumulator.device]
            for index, (label, x) in enumerate(shared_inputs()):
                record = shared_report["cases"][index]
                if record["input"] != label or record["input_bf16_sha256"] != digest(bf16_bits(x)):
                    raise ValueError("Generated shared input differs from the original report")
                checks = check(expert_stages(shared.forward(x), native=True), record["expected_sha256"])
                cases.append({"probe": "shared", "input": label, "checks": checks})
            if index + 1 != len(shared_report["cases"]):
                raise ValueError("Shared reference case count differs")
            for index, case in enumerate(aggregation_inputs()):
                record = aggregation_report["cases"][index]
                if (record["input"] != case["input"] or record["indices"] != case["indices"].tolist()
                        or record["shared_sha256"] != digest(case["shared"])
                        or record["contribution_sha256"] != {str(e): digest(a) for e, a in case["contributions"].items()}):
                    raise ValueError("Generated contributions differ from the original report")
                actual = accumulator.run(case["indices"], case["contributions"], case["shared"])
                cases.append({"probe": "aggregation", "input": case["input"], "checks": check(actual, record["expected_sha256"])})
            if index + 1 != len(aggregation_report["cases"]):
                raise ValueError("Aggregation reference case count differs")
            for index, case in enumerate(original_expert_inputs()):
                record = original_report["cases"][index]
                x, ids, weights = case["x"], case["indices"], case["weights"]
                if (record["input"] != case["input"] or record["input_bf16_sha256"] != digest(bf16_bits(x))
                        or record["indices"] != ids.tolist() or record["routing_weights"] != weights.tolist()):
                    raise ValueError("Generated original-expert input/routes differ from the reference")
                checks, contributions = {}, {}
                for expert in range(6):
                    route = float(weights[0, np.where(ids[0] == expert)[0][0]])
                    actual = expert_stages(experts[expert].forward(x, route), native=True)
                    checks.update({f"expert_{expert}.{name}": value for name, value in
                                   check(actual, record["experts"][expert]["expected_sha256"]).items()})
                    contributions[expert] = actual["w2"].reshape(1, HIDDEN)
                actual_shared = expert_stages(shared.forward(x), native=True)
                checks.update({f"shared.{name}": value for name, value in
                               check(actual_shared, record["shared_expected_sha256"]).items()})
                actual = accumulator.run(ids, contributions, actual_shared["w2"].reshape(1, HIDDEN))
                checks.update({f"aggregation.{name}": value for name, value in
                               check(actual, record["aggregation_expected_sha256"]).items()})
                cases.append({"probe": "original_experts", "input": case["input"], "checks": checks})
            if index + 1 != len(original_report["cases"]):
                raise ValueError("Original-expert reference case count differs")
            cubins = {"routed": experts[0].cubin_sha256, "shared": shared.fp8_cubin_sha256,
                      "aggregation": accumulator.cubin_sha256}
            peaks = {"routed_experts": [e.device.peak for e in experts], "shared": shared.device.peak,
                     "aggregation": accumulator.device.peak}
    finally:
        budget.release(512 * MiB)
    torch_imported = any(n == "torch" or n.startswith("torch.") for n in sys.modules)
    for case in cases:
        case["status"] = "pass" if all(v == "pass" for v in case["checks"].values()) else "fail"
    cleanup = sum(sum(device.allocations.values()) for device in devices)
    passed = bool(cases) and all(c["status"] == "pass" for c in cases) and not torch_imported and cleanup == 0
    result = {"probe": "native_moe_arithmetic_recheck", "status": "pass" if passed else "fail", "cases": cases,
              "reference_report_sha256": {"shared-expert.json": shash, "moe-aggregation.json": ahash,
                                          "original-expert-accumulation.json": ohash},
              "numpy_version": np.__version__, "torch_installed": importlib.util.find_spec("torch") is not None,
              "torch_imported": torch_imported, "compiled_cubin_sha256": cubins,
              "peak_explicit_device_allocation_by_component": peaks,
              "cleanup_explicit_device_bytes": cleanup, "cleanup_host_reservation_bytes": budget.reserved + accumulator.budget.reserved,
              "host_probe_reservation_bytes": budget.peak + accumulator.budget.peak,
              "host_available_floor_bytes": budget.available_floor,
              "scope": "Rechecks supplied-input original-reference goldens; original gate/offload/model quality are separate gates"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "conditions": len(cases), "torch_installed": result["torch_installed"],
                      "torch_imported": torch_imported, "cleanup_explicit_device_bytes": cleanup}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
