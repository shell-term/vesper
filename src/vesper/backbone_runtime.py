"""Bind bounded native all-layer probes to immutable prepared original assets."""

from pathlib import Path
import numpy as np

from .attention import NativeAttention
from .attention_topology import KV_SOURCES, INDEX_SOURCES, BACKBONE_COMPRESSED_LAYERS
from .backbone import NativeBackbone, LAYERS
from .compression import NativeCompression
from .engram import NativeEngram
from .engram_assets import load_controls
from .engram_rows import EngramRows
from .hc import NativeHC
from .indexer import NativeIndexer
from .model_artifacts import digest
from .ngram import NgramState
from .normalization import NativeNormRoPE
from .output import NativeOutput
from .output_assets import EMBED_IDS, TENSOR_SHA256
from .projections import NativeProjections
from .sampling import NativeSampler
from .shared_attention import SharedAttentionFrame
from .tokenizer import PinnedTokenizer, compressed_token_map
from .window_attention import NativeWindowAttention


def embeddings(root):
    raw = (Path(root) / "output-fixture/embedding-rows.bin").read_bytes()
    if digest(raw) != TENSOR_SHA256["embedding_rows"]: raise ValueError("Original embedding rows changed")
    values = np.frombuffer(raw, np.uint16).reshape(len(EMBED_IDS), 5120)
    return {token: values[i] for i, token in enumerate(EMBED_IDS)}


def moe_files(root, bindings, layer):
    if type(layer) is not int or layer not in LAYERS: raise ValueError("Unregistered original backbone layer")
    return tuple(Path(root) / bindings["layers"][str(layer)][part]["path"] for part in ("gate", "router", "shared"))


def validate_moe_manifests(moe, bindings, layer):
    if any(getattr(moe, part + "_manifest") != bindings["layers"][str(layer)][part]["manifest"]
            for part in ("gate", "router", "shared")):
        raise ValueError("Original gate/router/shared payload binding differs")


def open_backbone(stack, root, arrays, token_rows, moe, head_arrays, *, temperature=.7):
    root = Path(root)
    hc = {l: stack.enter_context(NativeHC(arrays, layer=l, profile="backbone_probe")) for l in LAYERS}
    ps = {l: stack.enter_context(NativeProjections(arrays, layer=l, profile="backbone_probe")) for l in LAYERS}
    norm = stack.enter_context(NativeNormRoPE())
    sparse = stack.enter_context(NativeAttention(root / "native-attention-v1", profile="long_context"))
    comp = stack.enter_context(NativeCompression(arrays, layers=KV_SOURCES))
    index = stack.enter_context(NativeIndexer(arrays, layers=INDEX_SOURCES))
    windows = {l: stack.enter_context(NativeWindowAttention(arrays, ps[l], norm, sparse, profile="backbone_probe")) for l in (0, 1)}
    frame = stack.enter_context(SharedAttentionFrame(arrays, {l: ps[l] for l in BACKBONE_COMPRESSED_LAYERS}, norm, comp, index, sparse,
        BACKBONE_COMPRESSED_LAYERS, binding_profile="backbone_probe"))
    engrams, engram_manifests = {}, {}
    for l in (1, 14):
        a, manifest = load_controls(root / f"engram-fixture/controls{l}", l)
        rows = stack.enter_context(EngramRows(root / f"engram-fixture/rows/layer{l}", l))
        engrams[l] = stack.enter_context(NativeEngram(a, rows))
        engram_manifests[str(l)] = {"controls": manifest, "rows": rows.manifest}
    mapping, map_identity = compressed_token_map(PinnedTokenizer(root / "import-source"))
    sampler = stack.enter_context(NativeSampler(2026))
    head = stack.enter_context(NativeOutput(head_arrays["head.weight"]))
    engine = stack.enter_context(NativeBackbone(arrays, token_rows, hc, norm, windows, frame, moe, engrams, NgramState(mapping), sampler,
        head, head_arrays["norm.weight"], temperature=temperature))
    engine.probe_operators = {"projections": ps, "hc": hc, "norm": norm, "compressor": comp, "indexer": index, "sparse": sparse}
    engine.tokenizer_identity = map_identity
    engine.engram_manifests = engram_manifests
    return engine
