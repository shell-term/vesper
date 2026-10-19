"""Pinned metadata, original tensor addresses and a non-downloading import plan."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import urllib.request

from .checkpoint import MODEL, REVISION, LIMIT, read_small, read_range
from .resources import MemoryBudget, MiB, GiB

# Raw source bytes, rather than repository defaults or reconstructed configs.
FILES = {
    "config.json": ("config.json", 3311, "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"),
    "inference/config.json": ("inference-config.json", 1982, "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"),
    "model.safetensors.index.json": ("model.safetensors.index.json", 7470294, "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"),
    "tokenizer.json": ("tokenizer.json", 6367257, "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"),
    "tokenizer_config.json": ("tokenizer_config.json", 801, "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"),
    "inference/engram.py": ("engram.py", 8138, "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"),
    "LICENSE": ("LICENSE", 1084, "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040"),
}
CATALOG_SHA256 = "f2595189d8a80ca4e1245a3cbfbaa32920ee1a8daee71376f81c3ac8f4d7e4b9"
SHARD_INVENTORY_SHA256 = "3b5f61a86843940821622cc825b942be51eca3a2bae4b3167c75b5ec870a6406"
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4,
    "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1}


def digest(data): return hashlib.sha256(data).hexdigest()


def canonical(data): return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def shard_inventory(root):
    path=Path(root)/'tree.json'
    if path.stat().st_size>LIMIT: raise ValueError('Excessive pinned shard inventory')
    data=json.loads(path.read_bytes())
    records=[{'file':r['path'],'bytes':r['size'],'sha256':r['lfs']['oid']} for r in data if r['path'].endswith('.safetensors')]
    if canonical(records)!=SHARD_INVENTORY_SHA256 or len(records)!=48: raise ValueError('Original shard byte counts or SHA-256 identities differ')
    return records


def read_checked(root, name):
    filename, size, sha = FILES[name]; path = Path(root) / filename
    if path.stat().st_size != size: raise ValueError("Pinned model artifact byte count differs: " + name)
    raw = path.read_bytes()
    if digest(raw) != sha: raise ValueError("Pinned model artifact SHA-256 differs: " + name)
    return raw


def fetch_metadata(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    budget = MemoryBudget(cap=64 * MiB); budget.acquire(budget.cap)
    fetched = 0
    try:
        for name, (filename, size, sha) in FILES.items():
            path = root / filename
            if path.exists(): read_checked(root, name)
            else:
                raw = read_small(name)
                if len(raw) != size or digest(raw) != sha: raise ValueError("Unqualified model artifact: " + name)
                with path.open("xb") as file: file.write(raw)
                fetched += len(raw)
        tree=root/'tree.json'
        if not tree.exists():
            with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true&expand=false',timeout=45) as response:
                raw=response.read(LIMIT+1)
            if len(raw)>LIMIT: raise ValueError('Excessive pinned shard inventory')
            with tree.open('xb') as file: file.write(raw)
        shard_inventory(root)
        return {"model": MODEL, "revision": REVISION, "new_metadata_bytes": fetched,
            "files": {n: {"bytes": s, "sha256": h} for n, (_, s, h) in FILES.items()}}
    finally: budget.release(budget.cap)


def fetch_headers(root, progress=None):
    root = Path(root); fetch_metadata(root)
    index = json.loads(read_checked(root, "model.safetensors.index.json"))
    directory = root / "headers"; directory.mkdir(exist_ok=True)
    shards = sorted(set(index["weight_map"].values()))
    if shards != [f"model-{i:05}-of-00048.safetensors" for i in range(1, 49)]: raise ValueError("Checkpoint shard set differs")
    budget = MemoryBudget(cap=64 * MiB); budget.acquire(budget.cap)
    def fetch(shard):
        path = directory / (shard + ".json")
        if not path.exists():
            size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
            if not 0 < size <= LIMIT: raise ValueError("Excessive original shard header")
            raw = read_range(shard, 8, size); json.loads(raw)
            with path.open("xb") as file: file.write(raw)
        return shard
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for count, shard in enumerate(pool.map(fetch, shards), 1):
                if progress: progress({"headers": count, "shard": shard})
    finally: budget.release(budget.cap)


def load_catalog(root, *, require_qualified=True):
    budget = MemoryBudget(cap=512 * MiB); budget.acquire(budget.cap)
    try:
        root = Path(root)
        for name in FILES: read_checked(root, name)
        index = json.loads(read_checked(root, "model.safetensors.index.json"))
        records, headers = {}, []
        for shard in sorted(set(index["weight_map"].values())):
            if not re.fullmatch(r"model-\d{5}-of-00048\.safetensors", shard): raise ValueError("Unexpected shard name")
            path = root / "headers" / (shard + ".json")
            if not 0 < path.stat().st_size <= LIMIT: raise ValueError("Excessive original header")
            raw = path.read_bytes(); header = json.loads(raw); start = len(raw) + 8
            entries = sorted(((n,t) for n,t in header.items() if n != "__metadata__"), key=lambda p:p[1]["data_offsets"])
            end = 0
            for name, t in entries:
                shape, dtype = t["shape"], t["dtype"]
                if (dtype not in DTYPE_BYTES or any(type(d) is not int or d <= 0 for d in shape)
                        or t["data_offsets"] != [end, end + math.prod(shape) * DTYPE_BYTES[dtype]]
                        or index["weight_map"].get(name) != shard or name in records):
                    raise ValueError("Original tensor dtype, shape, contiguous extent or shard assignment differs")
                size = math.prod(shape) * DTYPE_BYTES[dtype]
                records[name] = {"name":name, "shard":shard, "dtype":dtype, "shape":shape,
                    "shard_byte_offset":start+end, "bytes":size}
                end += size
            headers.append({"shard":shard, "header_bytes":len(raw), "header_sha256":digest(raw), "tensor_count":len(entries), "file_bytes":start+end})
        if set(records) != set(index["weight_map"]) or sum(r["bytes"] for r in records.values()) != index["metadata"]["total_size"]:
            raise ValueError("Incomplete original artifact catalog")
        identity = canonical({"tensors":records, "headers":headers})
        if require_qualified and (not CATALOG_SHA256 or identity != CATALOG_SHA256): raise ValueError("Unqualified original artifact catalog identity")
        inventory={s['file']:s for s in shard_inventory(root)}
        if any(h['file_bytes']!=inventory[h['shard']]['bytes'] for h in headers): raise ValueError('Original shard extent differs from released LFS size')
        for h in headers: h['original_file_sha256']=inventory[h['shard']]['sha256']
        return records, {"schema_version":1, "model":MODEL, "revision":REVISION, "catalog_sha256":identity,
            "tensor_count":len(records), "original_tensor_bytes":sum(t["bytes"] for t in records.values()), "shards":headers,
            "metadata":{n:{"bytes":s,"sha256":h} for n,(_,s,h) in FILES.items()}, "host_allowance_bytes":budget.peak,
            "reservation_is_RSS_limit":False, "weight_payload_bytes_fetched":0}
    finally: budget.release(budget.cap)


def full_import_plan(root, target, *, disk_reserve=64 * GiB):
    if type(disk_reserve) is not int or disk_reserve < 32 * GiB: raise ValueError("Keep an explicit disk reserve of at least 32 GiB")
    records, catalog = load_catalog(root)
    usage = shutil.disk_usage(Path(target).resolve())
    shard_bytes = sum(h["file_bytes"] for h in catalog["shards"])
    metadata_parts={'pinned_source_files':sum(s for _,s,_ in FILES.values()),
        'detached_shard_headers':sum(h['header_bytes'] for h in catalog['shards']),
        'shard_inventory':(Path(root)/'tree.json').stat().st_size}
    metadata_bytes = sum(metadata_parts.values())
    # One original copy; no lossless-to-expanded conversion or second full cache.
    integrity_index_bytes=sum((((h['file_bytes']+65535)//65536)*32+4095)//4096*4096 for h in catalog['shards'])
    required = shard_bytes + metadata_bytes + integrity_index_bytes + disk_reserve
    groups = {"Engram_tables":0, "backbone_experts":0, "backbone_other":0, "MTP":0, "vision":0}
    for name, t in records.items():
        group = ("Engram_tables" if ".engram.embed." in name else "MTP" if name.startswith("mtp.") else
            "vision" if name.startswith("vision.") else "backbone_experts" if ".ffn.experts." in name else "backbone_other")
        groups[group] += t["bytes"]
    return {"schema_version":1, "probe":"original_artifact_import_plan", "status":"planned", "model":MODEL,"revision":REVISION,
        "catalog":catalog, "tensor_bytes_by_category":groups, "original_shard_file_bytes":shard_bytes,"metadata_bytes":metadata_bytes,
        "metadata_bytes_by_kind":metadata_parts,
        "integrity_index_bytes":integrity_index_bytes,"integrity_block_bytes":65536,"shard_inventory_sha256":SHARD_INVENTORY_SHA256,
        "target_volume_total_bytes":usage.total,"target_volume_available_bytes":usage.free,"disk_reserve_bytes":disk_reserve,
        "required_free_bytes":required,"capacity_sufficient":usage.free>=required,"additional_free_bytes_required":max(0,required-usage.free),
        "full_import_authorized_or_started":False,"full_checkpoint_downloaded":False,"weight_payload_bytes_fetched":0,
        "storage_policy":"one original shard copy; direct required-row reads; no model-wide mmap or expanded checkpoint duplicate",
        "provisional_full_runtime_host_cap_bytes":96*GiB,"system_available_floor_bytes":32*GiB,
        "whole_runtime_memory_governor":"pending_issue9","full_model_runtime":"pending_issue10"}
