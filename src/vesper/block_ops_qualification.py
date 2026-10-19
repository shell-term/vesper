"""Audit Issue #7's original acceptance against sealed original/native evidence."""

import hashlib
from importlib import import_module
import json
from pathlib import Path

from .checkpoint import MODEL, REVISION
from .attention_topology import GROUPS, role
from .resources import MemoryBudget, GiB, MiB, physical_memory

RESULT_ROOT = "results/2026-10-19-block-ops-qualification"
QUALIFIED_MANIFEST_SHA256 = "cb8c4ad28d89d5784a5f93fb7b538813eaacd0fd8941d80d455387448635851a"
EVIDENCE = (
    ("sparse_attention", "attention_validation", "results/2026-10-18-native-attention/attention.json"),
    ("window_kv", "kv_validation", "results/2026-10-18-window-kv/window-kv.json"),
    ("normalization_rope", "normalization_validation", "results/2026-10-18-normalization-rope/normalization.json"),
    ("window_projections", "projection_validation", "results/2026-10-18-attention-projections/projections.json"),
    ("layer0_hc_block", "block_validation", "results/2026-10-18-layer0-block/block.json"),
    ("compression", "compression_validation", "results/2026-10-18-compression/compression.json"),
    ("indexer", "indexer_validation", "results/2026-10-18-indexer/indexer.json"),
    ("compressed_attention", "compressed_attention_validation", "results/2026-10-19-compressed-attention/compressed-attention.json"),
    ("long_context", "long_context_validation", "results/2026-10-19-long-context/long-context.json"),
    *(("topology_" + "_".join(map(str, g)), "topology_validation", "results/2026-10-19-attention-topology/group-" + "-".join(map(str, g)) + ".json") for g in GROUPS),
    ("layer2_block", "compressed_block_validation", "results/2026-10-19-compressed-block/block.json"),
    ("pair_2_3", "block_pair_validation", "results/2026-10-19-block-pair/pair.json"),
    ("pair_20_21", "block_pair_validation", "results/2026-10-19-ratio1-block-pair/pair.json"),
    ("refresh_chain", "block_chain_validation", "results/2026-10-19-refresh-block-chain/chain.json"),
)
CRITERIA = (
    ("position_kv_sharing_selection", "Preserve rotary/position behavior, compressed and sliding-window KV state, source-layer sharing and sparse index selection."),
    ("quantization_rollback", "Preserve original activation/KV quantization and rollback/state semantics."),
    ("rms_hc_residuals", "Qualify RMS normalization, Hyper-Connection mixes/Sinkhorn and residual boundaries against the pinned source."),
    ("bounded_intermediate_gates", "Use bounded per-component fixtures and publish exact intermediate comparisons before connecting the full block."),
    ("prefill_decode_scope", "Cover chunk/prefill/decode boundaries; keep batching and distributed scope explicit."),
)
TOPOLOGY_IDS = tuple(n for n, _, _ in EVIDENCE if n.startswith("topology_"))
CHECK_EVIDENCE = {
    "position_kv_sharing_selection": ("sparse_attention", "window_kv", "normalization_rope", "window_projections", "compression", "indexer", "compressed_attention", "long_context", *TOPOLOGY_IDS, "refresh_chain"),
    "quantization_rollback": ("window_kv", "compression", "indexer", "compressed_attention", "long_context", *TOPOLOGY_IDS, "layer0_hc_block", "layer2_block", "pair_2_3", "pair_20_21", "refresh_chain"),
    "rms_hc_residuals": ("normalization_rope", "layer0_hc_block", "layer2_block", "pair_2_3", "pair_20_21", "refresh_chain"),
    "bounded_intermediate_gates": tuple(n for n, _, _ in EVIDENCE),
    "prefill_decode_scope": ("window_kv", "window_projections", "compression", "compressed_attention", "long_context", *TOPOLOGY_IDS, "refresh_chain"),
}


def manifest_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_manifest(data):
    if (data.get("schema_version"), data.get("probe"), data.get("model"), data.get("revision")) != (1, "issue7_evidence_manifest", MODEL, REVISION):
        raise ValueError("Block-operations evidence manifest identity differs")
    if not QUALIFIED_MANIFEST_SHA256 or manifest_digest(data) != QUALIFIED_MANIFEST_SHA256:
        raise ValueError("Unqualified or changed block-operations evidence manifest")
    rows = data.get("evidence", [])
    if [(r.get("id"), r.get("validator"), r.get("reference", {}).get("file")) for r in rows] != list(EVIDENCE):
        raise ValueError("Block-operations evidence corpus is incomplete or substituted")
    if data.get("criteria") != [{"id": n, "criterion": text, "evidence": list(CHECK_EVIDENCE[n])} for n, text in CRITERIA]:
        raise ValueError("Original Issue #7 acceptance criteria or supporting gates changed")
    for row in rows:
        if row.get("native_recheck", {}).get("file") != f"{RESULT_ROOT}/rechecks/{row['id']}.json":
            raise ValueError("Unexpected native evidence path")
        for artifact in (row["reference"], row["native_recheck"]):
            if (set(artifact) != {"file", "bytes", "sha256"} or type(artifact["bytes"]) is not int or not 0 < artifact["bytes"] <= 64 * MiB
                    or type(artifact["sha256"]) is not str or len(artifact["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in artifact["sha256"])):
                raise ValueError("Bounded evidence artifact descriptor differs")
    return rows


def read_verified(root, artifact):
    root = Path(root).resolve(); path = (root / artifact["file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != artifact["bytes"]:
        raise ValueError("Evidence path or actual byte count differs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]: raise ValueError("Evidence artifact SHA-256 differs")
    return json.loads(raw)


def qualify(root, manifest_path, progress=None):
    path = Path(manifest_path)
    if path.stat().st_size > 128 * 1024: raise ValueError("Excessive evidence manifest")
    manifest = json.loads(path.read_bytes()); evidence = validate_manifest(manifest)
    budget = MemoryBudget(cap=2 * GiB); budget.acquire(budget.cap)
    checks, minimum = [], physical_memory()["available"]
    try:
        for row in evidence:
            budget.acquire(0)
            report = read_verified(root, row["reference"])
            import_module("vesper." + row["validator"]).validate_goldens(report)
            del report
            native = read_verified(root, row["native_recheck"])
            if (native.get("status") != "pass" or native.get("reference_report_sha256") != row["reference"]["sha256"]
                    or any(native.get(n + "_" + s) is not False for n in ("torch", "tilelang") for s in ("installed", "imported"))):
                raise ValueError("Independent native replay or its exact original report identity differs")
            checks.append({"id": row["id"], "status": "pass", "original_goldens_validated": True,
                "reference": row["reference"], "native_recheck": row["native_recheck"], "framework_free_replay": True})
            del native
            minimum = min(minimum, physical_memory()["available"])
            if progress: progress({"evidence": row["id"], "status": "pass"})
        if minimum < budget.available_floor: raise MemoryError("Evidence audit fell below the system available-memory floor")
    finally: budget.release(budget.cap)
    return {"schema_version": 1, "probe": "issue7_block_operations_acceptance", "status": "pass", "model": MODEL, "revision": REVISION,
        "evidence_manifest_sha256": manifest_digest(manifest), "qualified_evidence_groups": len(checks), "evidence_checks": checks,
        "acceptance_checks": [{"id": n, "criterion": text, "status": "pass", "evidence": list(CHECK_EVIDENCE[n])} for n, text in CRITERIA],
        "configured_backbone_attention_roles": [{"layer": l, "ratio": role(l).ratio, "KV_owner": role(l).kv_owner,
            "index_owner": role(l).index_owner, "candidate_owner": role(l).candidate_owner} for l in range(40)],
        "scope": {"world_size": 1, "Attention_batch_sizes": [1, 2], "Attention_prefill_queries": [1, 2, 3],
            "maximum_continuous_empty_start_Attention_context": 1152, "maximum_synthetic_Attention_context": 16392,
            "long_context_Attention_layers": [2, 20, 24], "candidate_pruning_Attention_layers": [20, 24],
            "named_role_Attention_layers": [l for group in GROUPS for l in group],
            "complete_refresh_chain": [20, 21, 22, 23, 24, 25], "complete_refresh_chain_batch_size": 1,
            "complete_refresh_chain_queries_per_step": 1, "complete_refresh_chain_unprimed_context": 4,
            "entry_hidden_states": "generated except the separately captured original layer-0 prefix cases",
            "all_forty_original_layers_numerically_executed": False, "distributed_execution_qualified": False,
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "qualified_oracle": "explicit_original_arithmetic_PTX_memory_adapter"},
        "memory": {"host_allowance_bytes": budget.peak, "host_reserved_after_bytes": budget.reserved,
            "available_floor_bytes": budget.available_floor, "minimum_sampled_available_bytes": minimum, "reservation_is_RSS_limit": False},
        "audit_kind": "sealed_artifact_and_original_golden_validation; GPU comparisons are the linked native/reference runs",
        "issue7_original_acceptance_complete": True, "M2_complete": False,
        "remaining_issues": [8, 9, 10, 11], "full_model_quality": "not_run", "text_generation_available": False, "end_to_end_speed": "not_measured"}
