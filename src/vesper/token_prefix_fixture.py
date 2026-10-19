"""Additional original layer-1 bindings for the bounded four-block prefix."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .block_fixture import fetch_block_fixture, load_block_fixture
from .gate import fetch_gate_fixture, load_gate_fixture
from .router import fetch_router_fixture, load_router_fixture
from .shared import fetch_shared_fixture, load_shared_fixture
from .topology_fixture import fetch_layer, load_layer


def prepare_layer1(directory, progress=None):
    root = Path(directory)
    jobs = [
        ("attention", lambda p: fetch_layer(p, 1, profile="token_prefix"),
         lambda p: load_layer(p, 1, profile="token_prefix"), None),
        ("hc-norm", lambda p: fetch_block_fixture(p, layer=1, profile="token_prefix"),
         lambda p: load_block_fixture(p, layer=1, profile="token_prefix"), None),
        ("gate", lambda p: fetch_gate_fixture(p, layer=1), load_gate_fixture, "native-gate.npz"),
        ("router", lambda p: fetch_router_fixture(p, layer=1), load_router_fixture, "native-router.npz"),
        ("shared", lambda p: fetch_shared_fixture(p, layer=1), load_shared_fixture, "native-shared.npz"),
    ]
    def prepare(job):
        name, fetch, load, file = job
        path = root / name
        if path.exists():
            arrays, manifest = load(path if file is None else path / file)
            del arrays
        else:
            manifest = fetch(path)
        if manifest.get("layer") != 1:
            raise ValueError("Token prefix artifact does not bind original layer 1")
        return name, manifest
    results = {}
    # Only two bounded fetchers; completed arrays are never queued in futures.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, job) for job in jobs]
        for future in as_completed(futures):
            name, manifest = future.result()
            results[name] = manifest
            if progress:
                progress({"prepared_layer": 1, "component": name, "original_bytes": manifest["weights_fetched_bytes"]})
    return {"probe": "original_layer1_token_prefix_artifacts", "layer": 1,
        "components": results, "full_checkpoint_downloaded": False}
