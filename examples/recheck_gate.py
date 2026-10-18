"""Recheck published gate goldens with NumPy/CUDA only; do not import PyTorch."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from vesper.gate import GateProjection, load_gate_fixture
from vesper.gate_reference import gate_inputs
from vesper.router import RouterProbe, load_router_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--router-fixture", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.reference_report.stat().st_size > 8 * 1024**2:
        raise ValueError("Reference report exceeds the bounded example size")
    raw = args.reference_report.read_bytes()
    expected = json.loads(raw)
    weight, fixture = load_gate_fixture(args.fixture)
    biases, router_fixture = load_router_fixture(args.router_fixture)
    if (expected.get("probe") != "gate_projection_parity" or expected.get("status") != "pass"
            or expected.get("native_candidate_qualified") is not True
            or expected["fixture"] != fixture or expected["router_fixture"] != router_fixture):
        raise ValueError("Expected a passing gate report with these exact original fixtures")
    cases = []
    with GateProjection(weight, args.device) as gate, RouterProbe(biases, args.device) as router:
        for metadata, x in gate_inputs(weight):
            logits = gate.project(x)
            for image_span in (False, True):
                position = len(cases)
                reference = expected["cases"][position]
                if (any(reference[name] != value for name, value in metadata.items())
                        or reference["image_span"] != image_span
                        or reference["input_sha256"] != hashlib.sha256(memoryview(x)).hexdigest()):
                    raise ValueError("Generated input differs from the reference; check the NumPy/source versions")
                actual = router.inspect(logits, image_span)
                logit_hash = hashlib.sha256(memoryview(logits)).hexdigest()
                indices = actual["indices"].tolist()
                weights = actual["weights"].view(np.uint32).tolist()
                passed = (logit_hash == reference["reference_logits_sha256"] and indices == reference["reference_indices"]
                          and weights == reference["reference_weight_bits"])
                cases.append({**metadata, "image_span": image_span, "logits_sha256": logit_hash,
                              "indices": indices, "weight_bits": weights, "status": "pass" if passed else "fail"})
        library_version, projection_peak, router_peak = gate.blas.version, gate.device.peak, router.device.peak
    torch_imported = any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    passed = len(cases) == len(expected["cases"]) and all(case["status"] == "pass" for case in cases) and not torch_imported
    result = {"probe": "native_gate_recheck", "status": "pass" if passed else "fail", "cases": cases,
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "numpy_version": np.__version__,
              "torch_installed": importlib.util.find_spec("torch") is not None, "torch_imported": torch_imported,
              "cublas_version": library_version, "projection_peak_explicit_device_bytes": projection_peak,
              "router_peak_explicit_device_bytes": router_peak,
              "cleanup_projection_explicit_device_bytes": sum(gate.device_object.allocations.values()),
              "cleanup_router_explicit_device_bytes": sum(router.device_object.allocations.values()),
              "cleanup_host_reservation_bytes": gate.host_budget.reserved,
              "scope": "Rechecks published supplied-input gate goldens; no connected MoE/model or speed claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "conditions": len(cases), "torch_installed": result["torch_installed"],
                      "torch_imported": torch_imported, "cublas_version": library_version}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
