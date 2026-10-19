"""Original bindings for all forty layers; no full-shard/model download."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil

from .block_fixture import fetch_block_fixture, load_block_fixture
from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION
from .engram_assets import load_controls
from .gate import fetch_gate_fixture, load_gate_fixture
from .model_artifacts import CATALOG_SHA256, canonical
from .prefix import load_prefix_fixture
from .resources import GiB
from .router import fetch_router_fixture, load_router_fixture
from .shared import fetch_shared_fixture, load_shared_fixture
from .topology_fixture import fetch_layer, load_layer

LAYERS = tuple(range(40))


def controls(catalog):
    return {n: t for n, t in catalog.items() if (match := re.match(r"^layers\.(\d+)\.", n))
        and int(match[1]) < 40 and ".ffn.experts." not in n and ".engram.embed." not in n}


def _entries(manifest):
    return [manifest["tensor"]] if "tensor" in manifest else manifest["tensors"]


def verify_bindings(bindings, catalog):
    expected, found = controls(catalog), {}
    if set(bindings) != {str(l) for l in LAYERS}: raise ValueError("Missing original backbone layer bindings")
    for key, parts in bindings.items():
        layer = int(key)
        required = {"prefix", "gate", "router", "shared"} if layer == 0 else {"attention", "hc-norm", "gate", "router", "shared"}
        if layer in (1, 14): required.add("engram")
        if set(parts) != required: raise ValueError("Incomplete original layer components")
        for part, record in parts.items():
            path = Path(record["path"])
            if path.is_absolute() or ".." in path.parts: raise ValueError("Expected a bounded relative fixture path")
            manifest = record["manifest"]
            if (manifest.get("model"), manifest.get("revision"), manifest.get("layer")) != (MODEL, REVISION, layer):
                raise ValueError("Cross-model or cross-layer original binding")
            for entry in _entries(manifest):
                name = entry.get("tensor", entry["name"])
                if name not in expected or name in found:
                    raise ValueError("Unexpected or duplicated original control tensor")
                target = expected[name]
                if any(entry[k] != target[k] for k in ("dtype", "shape", "bytes", "shard", "shard_byte_offset")):
                    raise ValueError("Original tensor does not match authenticated catalog extent")
                if re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
                    raise ValueError("Missing original control payload identity")
                found[name] = entry
    if set(found) != set(expected) or len(found) != 1251 or sum(e["bytes"] for e in found.values()) != 7199083968:
        raise ValueError("All-forty-layer original control census differs")
    return found


def prepare(root, progress=None):
    root = Path(root).resolve()
    if shutil.disk_usage(root).free < 64 * GiB + 8 * GiB:
        raise OSError("Keep 64 GiB free plus the bounded all-layer control preparation")
    bindings = {str(l): {} for l in LAYERS}
    with CatalogReader(root / "import-source") as reader:
        def task(layer, part, old, file, fetch, load):
            target = old if old is not None and old.exists() else root / f"backbone-fixture/layer{layer}/{part}"
            existing = (target / file).exists() if file else target.exists()
            if existing:
                arrays, manifest = load(target / file if file else target); del arrays
                reused = True
            else:
                if shutil.disk_usage(root).free < 64 * GiB + 512 * 1024**2:
                    raise OSError("All-layer preparation would consume its disk reserve")
                manifest = fetch(target); reused = False
            return layer, part, {"path": target.relative_to(root).as_posix() + ("/" + file if file else ""), "manifest": manifest}, reused
        jobs = []
        for layer in LAYERS:
            if layer == 0:
                arrays, m = load_prefix_fixture(root / "prefix-fixture/native-prefix.npz"); del arrays
                bindings["0"]["prefix"] = {"path": "prefix-fixture/native-prefix.npz", "manifest": m}
            else:
                old = root / ("token-prefix-fixture/layer1/attention" if layer == 1 else f"topology-fixture/{layer}")
                jobs.append((layer, "attention", old, None,
                    lambda p, l=layer: fetch_layer(p, l, profile="backbone_probe", reader=reader),
                    lambda p, l=layer: load_layer(p, l, profile="backbone_probe")))
                old = root / ("token-prefix-fixture/layer1/hc-norm" if layer == 1 else
                    "compressed-block-fixture/hc-norm" if layer == 2 else f"block{layer}-fixture/hc-norm")
                jobs.append((layer, "hc-norm", old, None,
                    lambda p, l=layer: fetch_block_fixture(p, layer=l, profile="backbone_probe", reader=reader),
                    lambda p, l=layer: load_block_fixture(p, layer=l, profile="backbone_probe")))
            for part, fetch, load in (("gate", fetch_gate_fixture, load_gate_fixture), ("router", fetch_router_fixture, load_router_fixture),
                                     ("shared", fetch_shared_fixture, load_shared_fixture)):
                old = root / (f"{part}-fixture" if layer == 0 else f"token-prefix-fixture/layer1/{part}" if layer == 1 else
                    f"compressed-block-fixture/{part}" if layer == 2 else f"block{layer}-fixture/{part}")
                jobs.append((layer, part, old, f"native-{part}.npz", lambda p, l=layer, f=fetch: f(p, layer=l, reader=reader), load))
            if layer in (1, 14):
                arrays, m = load_controls(root / f"engram-fixture/controls{layer}", layer); del arrays
                bindings[str(layer)]["engram"] = {"path": f"engram-fixture/controls{layer}", "manifest": m}
        reused_count = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(task, *job) for job in jobs]
            for future in as_completed(futures):
                layer, part, record, reused = future.result()
                bindings[str(layer)][part] = record; reused_count += reused
                if progress: progress({"layer": layer, "part": part, "reused": reused, "prepared_components": sum(map(len, bindings.values())),
                    "new_payload_bytes": reader.stats()["payload_bytes_fetched"]})
        tensors = verify_bindings(bindings, reader.catalog)
        result = {"schema_version": 1, "probe": "original_all40_control_bindings", "status": "pass", "model": MODEL, "revision": REVISION,
            "catalog_sha256": CATALOG_SHA256, "layers": bindings, "control_tensor_count": len(tensors), "control_tensor_bytes": sum(e["bytes"] for e in tensors.values()),
            "reader": reader.stats(), "reused_components": reused_count, "workers": 4,
            "full_checkpoint_downloaded": False, "whole_shard_SHA256_registration_run": False, "full_model_execution_run": False}
        result["bindings_sha256"] = canonical(bindings)
    return result


def load(root, report):
    root = Path(root)
    if (report.get("probe"), report.get("status"), report.get("model"), report.get("revision"), report.get("catalog_sha256")) != (
            "original_all40_control_bindings", "pass", MODEL, REVISION, CATALOG_SHA256):
        raise ValueError("Unpinned all-layer control bindings")
    if canonical(report["layers"]) != report["bindings_sha256"]: raise ValueError("Changed all-layer binding receipt")
    with CatalogReader(root / "import-source") as reader: verify_bindings(report["layers"], reader.catalog)
    arrays = {}
    for layer in LAYERS:
        parts = report["layers"][str(layer)]
        for part in ("prefix",) if layer == 0 else ("attention", "hc-norm"):
            path = root / parts[part]["path"]
            a, m = (load_prefix_fixture(path) if part == "prefix" else load_layer(path, layer, profile="backbone_probe") if part == "attention"
                    else load_block_fixture(path, layer=layer, profile="backbone_probe"))
            if m != parts[part]["manifest"]: raise ValueError("Original all-layer fixture changed")
            arrays.update(a)
    return arrays
