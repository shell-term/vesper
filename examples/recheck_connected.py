"""Recheck the complete connected/failure matrix with no PyTorch dependency."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from vesper.connected import ConnectedConfig, ConnectedMoE
from vesper.connected_fixture import connected_inputs, load_connected_plan
from vesper.connected_reference import exercise_connected, native_connected_stages
from vesper.formats import bf16_bits
from vesper.resources import MiB
from vesper.shared_reference import digest
from vesper.storage import expert_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--gate-fixture", type=Path, required=True)
    parser.add_argument("--router-fixture", type=Path, required=True)
    parser.add_argument("--shared-fixture", type=Path, required=True)
    parser.add_argument("--captured-inputs", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--helper-device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_report.stat().st_size > 8 * MiB:
        raise ValueError("Reference report exceeds the bounded example size")
    raw = args.reference_report.read_bytes()
    report = json.loads(raw)
    plan = load_connected_plan(args.plan)
    if report.get("probe") != "connected_moe_parity" or report.get("status") != "pass" or report["plan"] != plan:
        raise ValueError("Expected a passing scoped reference report with this exact plan")
    rows = []
    for metadata, x in connected_inputs(args.captured_inputs):
        for image_span in (False, True):
            record = plan["cases"][len(rows)]
            if record["input"] != metadata["input"] or record["image_span"] != image_span or digest(bf16_bits(x)) != record["input_bf16_sha256"]:
                raise ValueError("Generated inputs differ from the published reference")
            rows.append((record, x))
    if len(rows) != len(plan["cases"]):
        raise ValueError("Input count differs from the reference")
    @contextmanager
    def factory(mode):
        config = ConnectedConfig()
        cache = replace(config.cache, primary_device=args.device, helper_device=args.helper_device)
        if mode == "two_slots_no_ram":
            cache = replace(cache, primary_slots=1, helper_slots=1, ram_cache_bytes=0)
        with ConnectedMoE(args.store, args.gate_fixture, args.router_fixture, args.shared_fixture, replace(config, cache=cache)) as runtime:
            if (runtime.gate_manifest != report["gate_fixture"] or runtime.router_manifest != report["router_fixture"]
                    or runtime.shared_manifest != report["shared_fixture"]):
                raise ValueError("Original gate/router/shared fixture provenance differs")
            expected = {expert_key(m["layer"], m["expert"]): m for m in report["expert_fixtures"]}
            actual = {key: entry["provenance"] for key, entry in runtime.experts.store.entries.items()}
            if actual != expected:
                raise ValueError("Native store and reference original tensor provenance differ")
            yield runtime
    def checker(index, result):
        actual = native_connected_stages(result)
        expected = report["reference_goldens"][index]["stage_sha256"]
        if set(actual) != set(expected):
            raise ValueError("Published boundary set differs")
        return [{"check": name, "status": "pass" if digest(array) == expected[name] else "fail"} for name, array in actual.items()]
    records, phases, faults = exercise_connected(factory, rows, checker)
    torch_imported = any(n == "torch" or n.startswith("torch.") for n in sys.modules)
    passed = (records and all(r["status"] == "pass" for r in records)
              and all(not any(p["cleanup"].values()) for p in phases) and not torch_imported)
    result = {"probe": "native_connected_moe_recheck", "status": "pass" if passed else "fail",
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "cases": records, "phases": phases, "faults": faults,
              "torch_installed": importlib.util.find_spec("torch") is not None, "torch_imported": torch_imported,
              "numpy_version": np.__version__, "input_source": plan["input_source"],
              "scope": "Same scoped connected/failure matrix; golden hashes; no new real-input/model-quality qualification"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({"status": result["status"], "successful_calls": len(records), "expected_failed_calls": len(faults),
                      "torch_installed": result["torch_installed"], "torch_imported": torch_imported}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
