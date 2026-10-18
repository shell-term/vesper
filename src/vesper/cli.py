from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vesper", description="Vesper native-precision MoE research probes")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "validate", "bench-transfer", "fetch-fixture", "bench-disk", "compare-reference",
                 "trace-reduction", "fetch-expert-fixture", "compare-expert", "build-store", "validate-cache",
                 "bench-cache", "validate-integrity", "fetch-router-fixture", "compare-router", "fetch-gate-fixture", "compare-gate",
                 "fetch-shared-fixture", "compare-shared", "compare-aggregation", "compare-original-accumulation",
                 "plan-connected", "prepare-connected-store", "compare-connected", "prefix-preflight", "fetch-prefix-fixture", "capture-prefix", "compare-attention-storage", "compare-attention-memory", "compare-native-attention", "compare-window-kv", "compare-normalization", "compare-projections", "compare-block", "fetch-compression-fixture", "compare-compression", "fetch-indexer-fixture", "compare-indexer"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path)
        if name == "fetch-indexer-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/indexer-fixture"))
        if name == "compare-indexer":
            command.add_argument("--fixture", type=Path, required=True)
            command.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
            command.add_argument("--device", type=int, default=0)
        if name == "fetch-compression-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/compression-fixture"))
        if name == "compare-compression":
            command.add_argument("--fixture", type=Path, required=True)
            command.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
            command.add_argument("--device", type=int, default=0)
        if name in ("validate", "bench-transfer", "validate-integrity"):
            command.add_argument("--devices", type=int, nargs="+")
        if name in ("validate", "compare-reference"):
            command.add_argument("--fixture", type=Path)
        if name in ("trace-reduction", "compare-expert", "compare-router", "compare-gate", "compare-shared"):
            command.add_argument("--fixture", type=Path, required=True)
        if name == "fetch-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/fixture"))
        if name == "fetch-expert-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/expert-fixture"))
            command.add_argument("--layer", type=int, default=0)
            command.add_argument("--expert", type=int, default=0)
        if name == "fetch-router-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/router-fixture"))
            command.add_argument("--layer", type=int, default=0)
        if name == "fetch-gate-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/gate-fixture"))
            command.add_argument("--layer", type=int, default=0)
        if name == "fetch-shared-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/shared-fixture"))
            command.add_argument("--layer", type=int, default=0)
        if name == "compare-original-accumulation":
            command.add_argument("--fixtures", type=Path, nargs=6, required=True)
            command.add_argument("--shared-fixture", type=Path, required=True)
        if name in ("plan-connected", "compare-connected"):
            command.add_argument("--gate-fixture", type=Path, required=True)
            command.add_argument("--router-fixture", type=Path, required=True)
            command.add_argument("--captured-inputs", type=Path)
        if name in ("prepare-connected-store", "compare-connected"):
            command.add_argument("--plan", type=Path, required=True)
            command.add_argument("--store", type=Path, required=True)
            command.add_argument("--expert-directory", type=Path, required=True)
        if name == "compare-connected":
            command.add_argument("--shared-fixture", type=Path, required=True)
            command.add_argument("--helper-device", type=int, default=1)
        if name in ("plan-connected", "build-store"):
            command.add_argument("--max-experts", type=int, default=48 if name == "plan-connected" else 4)
        if name == "fetch-prefix-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/prefix-fixture"))
        if name == "capture-prefix":
            command.add_argument("--fixture", type=Path, required=True)
            command.add_argument("--attention-reference", choices=("original", "ptx-memory"), default="original")
        if name == "compare-native-attention":
            command.add_argument("--bundle", type=Path, required=True, help="New local kernel bundle directory")
        if name in ("compare-window-kv", "compare-normalization", "compare-projections", "compare-block"):
            command.add_argument("--attention-bundle", type=Path, required=True)
        if name in ("compare-normalization", "compare-projections", "compare-block"):
            command.add_argument("--prefix-fixture", type=Path, required=True)
        if name == "compare-block":
            for argument in ("store", "gate-fixture", "router-fixture", "shared-fixture", "expert-directory"):
                command.add_argument("--" + argument, type=Path, required=True)
        if name in ("plan-connected", "compare-connected", "prefix-preflight", "capture-prefix", "compare-attention-storage", "compare-attention-memory", "compare-native-attention", "compare-window-kv", "compare-normalization", "compare-projections", "compare-block"):
            command.add_argument("--device", type=int, default=0)
        if name in ("compare-connected", "prefix-preflight", "capture-prefix", "compare-attention-storage", "compare-attention-memory", "compare-native-attention", "compare-window-kv", "compare-normalization", "compare-projections", "compare-block"):
            command.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
        if name == "compare-gate":
            command.add_argument("--router-fixture", type=Path, required=True)
            command.add_argument("--cublas-library", type=Path)
        if name in ("build-store", "validate-cache"):
            command.add_argument("--fixtures", type=Path, nargs="+", required=True)
        if name == "build-store":
            command.add_argument("--directory", type=Path, required=True)
        if name in ("validate-cache", "bench-cache"):
            command.add_argument("--store", type=Path, required=True)
            command.add_argument("--helper-device", type=int, default=1)
        if name == "validate-cache":
            command.add_argument("--integrity", choices=("readback", "gpu-sha256"), default="readback")
            command.add_argument("--profile", action="store_true")
        if name == "bench-cache":
            command.add_argument("--trials", type=int, default=6)
            command.add_argument("--repeats", type=int, default=8)
        if name == "bench-disk":
            command.add_argument("--directory", type=Path, default=Path("artifacts/disk-probe"))
            command.add_argument("--file", type=Path, help="Read an existing model file without creating scratch")
        if name in ("compare-reference", "compare-expert", "validate-cache", "compare-router", "compare-gate",
                    "compare-shared", "compare-aggregation", "compare-original-accumulation"):
            command.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
        if name in ("compare-reference", "compare-expert", "trace-reduction", "validate-cache", "bench-cache", "compare-router", "compare-gate",
                    "compare-shared", "compare-aggregation", "compare-original-accumulation"):
            command.add_argument("--device", type=int, default=0)
            if name != "bench-cache":
                command.add_argument("--reference-device", type=int)
        if name in ("compare-reference", "trace-reduction"):
            command.add_argument("--seed", type=int, default=20261018)
        if name == "compare-expert":
            command.add_argument("--seeds", type=int, nargs="+", default=[20261018])
            command.add_argument("--input-scales", type=float, nargs="+", default=[1.0])
            command.add_argument("--routing-weights", type=float, nargs="+", default=[0.25, 1.5, 0.0])
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from .cuda import Driver
            from .resources import physical_memory
            result = {"probe": "inventory", "status": "pass", "cuda": Driver().inventory(), "host_memory": physical_memory()}
        elif args.command == "validate":
            from .conformance import validate
            result = validate(args.devices, args.fixture)
        elif args.command == "bench-transfer":
            from .bandwidth import benchmark
            result = benchmark(args.devices)
        elif args.command == "fetch-fixture":
            from .checkpoint import fetch_fixture
            result = fetch_fixture(args.directory)
        elif args.command == "fetch-expert-fixture":
            from .checkpoint import fetch_expert_fixture
            result = fetch_expert_fixture(args.directory, args.layer, args.expert)
        elif args.command == "fetch-router-fixture":
            from .router import fetch_router_fixture
            result = fetch_router_fixture(args.directory, args.layer)
        elif args.command == "compare-router":
            from .router_reference import compare_router
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_router(args.fixture, args.reference_directory, args.device, reference)
        elif args.command == "fetch-gate-fixture":
            from .gate import fetch_gate_fixture
            result = fetch_gate_fixture(args.directory, args.layer)
        elif args.command == "compare-gate":
            from .gate_reference import compare_gate
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_gate(args.fixture, args.router_fixture, args.reference_directory,
                                  args.device, reference, args.cublas_library)
        elif args.command == "fetch-shared-fixture":
            from .shared import fetch_shared_fixture
            result = fetch_shared_fixture(args.directory, args.layer)
        elif args.command == "compare-shared":
            from .shared_reference import compare_shared
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_shared(args.fixture, args.reference_directory, args.device, reference)
        elif args.command == "compare-aggregation":
            from .moe_reference import compare_aggregation
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_aggregation(args.reference_directory, args.device, reference)
        elif args.command == "compare-original-accumulation":
            from .moe_reference import compare_original_accumulation
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_original_accumulation(args.fixtures, args.shared_fixture, args.reference_directory,
                                                  args.device, reference)
        elif args.command == "plan-connected":
            from .connected_fixture import plan_connected
            result = plan_connected(args.gate_fixture, args.router_fixture, args.captured_inputs,
                                    args.device, max_experts=args.max_experts)
        elif args.command == "prepare-connected-store":
            from .connected_fixture import prepare_connected_store
            result = prepare_connected_store(args.plan, args.expert_directory, args.store)
        elif args.command == "compare-connected":
            from .connected_reference import compare_connected
            result = compare_connected(args.plan, args.store, args.gate_fixture, args.router_fixture,
                                       args.shared_fixture, args.expert_directory, args.reference_directory,
                                       args.device, args.helper_device, args.captured_inputs)
        elif args.command == "prefix-preflight":
            from .prefix_reference import prefix_preflight
            result = prefix_preflight(args.reference_directory, args.device)
        elif args.command == "compare-attention-storage":
            from .attention_storage_reference import compare_attention_storage
            result = compare_attention_storage(args.reference_directory, args.device)
        elif args.command == "compare-attention-memory":
            from .attention_memory_reference import compare_attention_memory
            result = compare_attention_memory(args.reference_directory, args.device)
        elif args.command == "compare-native-attention":
            from .attention_reference import compare_native_attention
            result = compare_native_attention(args.reference_directory, args.bundle, args.device)
        elif args.command == "compare-window-kv":
            from .kv_reference import compare_window_kv
            result = compare_window_kv(args.reference_directory, args.attention_bundle, args.device)
        elif args.command == "compare-normalization":
            from .normalization_reference import compare_normalization
            result = compare_normalization(args.reference_directory, args.prefix_fixture, args.attention_bundle, args.device)
        elif args.command == "compare-projections":
            from .projection_reference import compare_projections
            result = compare_projections(args.reference_directory, args.prefix_fixture, args.attention_bundle, args.device)
        elif args.command == "fetch-indexer-fixture":
            from .indexer_fixture import fetch_indexer_fixture
            result = fetch_indexer_fixture(args.directory)
        elif args.command == "compare-indexer":
            from .indexer_reference import compare_indexer
            result = compare_indexer(args.reference_directory, args.fixture, args.device)
        elif args.command == "fetch-compression-fixture":
            from .compression_fixture import fetch_compression_fixture
            result = fetch_compression_fixture(args.directory)
        elif args.command == "compare-compression":
            from .compression_reference import compare_compression
            result = compare_compression(args.reference_directory, args.fixture, args.device)
        elif args.command == "compare-block":
            from .block_reference import compare_block
            result = compare_block(args.reference_directory, args.prefix_fixture, args.attention_bundle, args.store,
                                   args.gate_fixture, args.router_fixture, args.shared_fixture, args.expert_directory, args.device)
        elif args.command == "fetch-prefix-fixture":
            from .prefix import fetch_prefix_fixture
            result = fetch_prefix_fixture(args.directory)
        elif args.command == "capture-prefix":
            from .prefix_reference import capture_prefix
            result = capture_prefix(args.fixture, args.reference_directory, args.device, args.attention_reference)
        elif args.command == "build-store":
            from .storage import build_store
            result = build_store(args.fixtures, args.directory, max_experts=args.max_experts)
        elif args.command == "validate-cache":
            from .cache_reference import compare_cache
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_cache(args.store, args.fixtures, args.reference_directory,
                                   args.device, args.helper_device, reference,
                                   integrity_mode=args.integrity, profile=args.profile)
        elif args.command == "bench-cache":
            from .cache_benchmark import benchmark_cache
            result = benchmark_cache(args.store, args.device, args.helper_device, args.trials, args.repeats)
        elif args.command == "validate-integrity":
            from .integrity_validation import validate_integrity
            result = validate_integrity(args.devices)
        elif args.command == "bench-disk":
            from .disk import benchmark
            result = benchmark(args.directory, args.file)
        elif args.command == "trace-reduction":
            from .reduction import trace_reduction
            reference = 0 if args.reference_device is None else args.reference_device
            result = trace_reduction(args.fixture, args.device, reference, args.seed)
        elif args.command == "compare-expert":
            from .expert_reference import compare_expert
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_expert(args.fixture, args.reference_directory, args.device, reference,
                                    args.seeds, (None, *args.routing_weights), args.input_scales)
        else:
            from .reference import compare_official
            result = compare_official(args.fixture, args.reference_directory, args.device, args.reference_device, args.seed)
    except Exception as exc:
        result = {"probe": args.command, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    result["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered.encode("utf-8"))
    print(rendered, end="")
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
