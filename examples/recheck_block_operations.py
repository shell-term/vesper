"""Run fresh framework-free replays for the complete Issue #7 evidence set."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from vesper.block_ops_qualification import EVIDENCE


def replay_arguments(identity, fixtures):
    f = lambda n: str(fixtures / n)
    attention = ["--attention-bundle", f("native-attention-v1")]
    prefix = ["--prefix-fixture", f("prefix-fixture/native-prefix.npz")]
    compressed = ["--fixture", f("compressed-attention-fixture/native-compressed-attention.npz"),
        "--compression-fixture", f("compression-fixture/native-compression.npz"), "--indexer-fixture", f("indexer-fixture/native-indexer.npz"), *attention]
    if identity == "sparse_attention": return "recheck_attention.py", ["--bundle", f("native-attention-v1")]
    if identity == "window_kv": return "recheck_window_kv.py", attention
    if identity == "normalization_rope": return "recheck_normalization.py", [*prefix, *attention]
    if identity == "window_projections": return "recheck_projections.py", [*prefix, *attention]
    if identity == "layer0_hc_block": return "recheck_block.py", [*prefix, *attention, "--store", f("real-connected-store"),
        "--gate-fixture", f("gate-fixture/native-gate.npz"), "--router-fixture", f("router-fixture/native-router.npz"), "--shared-fixture", f("shared-fixture/native-shared.npz")]
    if identity == "compression": return "recheck_compression.py", ["--fixture", f("compression-fixture/native-compression.npz")]
    if identity == "indexer": return "recheck_indexer.py", ["--fixture", f("indexer-fixture/native-indexer.npz")]
    if identity == "compressed_attention": return "recheck_compressed_attention.py", compressed
    if identity == "long_context": return "recheck_long_context.py", compressed
    if identity.startswith("topology_"): return "recheck_topology.py", ["--fixture-directory", f("topology-fixture"), *attention]
    if identity == "layer2_block": return "recheck_compressed_block.py", ["--topology-fixtures", f("topology-fixture"),
        "--fixtures", f("compressed-block-fixture"), "--store", f("block2-store"), *attention]
    if identity in ("pair_2_3", "pair_20_21"):
        source, consumer = (2, 3) if identity == "pair_2_3" else (20, 21)
        return "recheck_block_pair.py", ["--topology-fixtures", f("topology-fixture"), *attention,
            "--source-fixtures", f("compressed-block-fixture" if source == 2 else "block20-fixture"),
            "--consumer-fixtures", f(f"block{consumer}-fixture"), "--source-store", f(f"block{source}-store"), "--consumer-store", f(f"block{consumer}-store")]
    if identity == "refresh_chain": return "recheck_block_chain.py", ["--topology-fixtures", f("topology-fixture"), "--block-root", str(fixtures), *attention]
    raise ValueError("Unregistered block-operations replay")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--fixtures-directory", type=Path, required=True)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--only", nargs="+", choices=[n for n, _, _ in EVIDENCE])
    a = p.parse_args()
    root, fixtures, output = a.repository.resolve(), a.fixtures_directory.resolve(), a.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [(n, path) for n, _, path in EVIDENCE if a.only is None or n in a.only]
    for identity, relative in selected:
        target, reference = output / (identity + ".json"), root / relative
        if target.exists(): raise FileExistsError("Refusing to replace a completed native replay: " + identity)
        script, extra = replay_arguments(identity, fixtures)
        with (output / (identity + ".log")).open("wb") as log:
            result = subprocess.run([sys.executable, str(root / "examples" / script), "--reference-report", str(reference),
                *extra, "--output", str(target)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError("Native replay failed; inspect its output log: " + identity)
        data = json.loads(target.read_bytes())
        if (data.get("status") != "pass" or data.get("reference_report_sha256") != hashlib.sha256(reference.read_bytes()).hexdigest()
                or any(data.get(n + "_" + s) is not False for n in ("torch", "tilelang") for s in ("installed", "imported"))):
            raise ValueError("Native replay evidence or framework independence differs: " + identity)
        print(json.dumps({"evidence": identity, "status": "pass"}), flush=True)
    print(json.dumps({"status": "pass", "selected_native_replays": len(selected)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
