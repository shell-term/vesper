"""Bounded fixture binding and operator lifetime for the 0..3 prefix probe."""

from pathlib import Path
import numpy as np

from .attention import NativeAttention
from .block_pair_validation import load_arrays as load_pair
from .compression import NativeCompression
from .engram import NativeEngram
from .engram_assets import load_controls
from .engram_rows import EngramRows
from .hc import NativeHC
from .indexer import NativeIndexer
from .model_artifacts import digest
from .ngram import NgramState
from .normalization import NativeNormRoPE
from .output_assets import EMBED_IDS, TENSOR_SHA256
from .prefix import load_prefix_fixture
from .projections import NativeProjections
from .sampling import NativeSampler
from .shared_attention import SharedAttentionFrame
from .token_prefix import NativeTokenPrefix, LAYERS
from .tokenizer import PinnedTokenizer, compressed_token_map
from .topology_fixture import load_layer
from .block_fixture import load_block_fixture
from .window_attention import NativeWindowAttention


def load_arrays(root):
    root = Path(root)
    arrays, m = load_prefix_fixture(root / "prefix-fixture/native-prefix.npz")
    manifests = {"prefix0": m}
    for part, load in (("attention", lambda p: load_layer(p, 1, profile="token_prefix")),
                       ("hc-norm", lambda p: load_block_fixture(p, layer=1, profile="token_prefix"))):
        a, m = load(root / "token-prefix-fixture/layer1" / part)
        arrays.update(a); manifests["layer1_" + part] = m
    a, m = load_pair(root / "topology-fixture", root / "compressed-block-fixture", root / "block3-fixture")
    arrays.update(a); manifests.update(m)
    raw = (root / "output-fixture/embedding-rows.bin").read_bytes()
    if digest(raw) != TENSOR_SHA256["embedding_rows"]:
        raise ValueError("Token prefix original embedding rows differ")
    embedding = np.frombuffer(raw, np.uint16).reshape(len(EMBED_IDS), 5120)
    return arrays, {token: embedding[i] for i, token in enumerate(EMBED_IDS)}, manifests


def control_folder(root, layer):
    if type(layer) is not int or layer not in LAYERS:
        raise ValueError("Unregistered token-prefix control layer")
    root = Path(root)
    return (root if layer == 0 else root / "token-prefix-fixture/layer1" if layer == 1 else
            root / "compressed-block-fixture" if layer == 2 else root / "block3-fixture")


def moe_files(root, layer):
    folder = control_folder(root, layer)
    if layer == 0:
        return tuple(Path(root) / f"{name}-fixture/native-{name}.npz" for name in ("gate", "router", "shared"))
    return tuple(folder / f"{name}/native-{name}.npz" for name in ("gate", "router", "shared"))


def open_prefix(stack, root, arrays, embeddings, moe, *, ordinal=0):
    root = Path(root)
    hc = {l: stack.enter_context(NativeHC(arrays, ordinal, layer=l, profile="token_prefix")) for l in LAYERS}
    ps = {l: stack.enter_context(NativeProjections(arrays, ordinal, layer=l, profile="token_prefix")) for l in LAYERS}
    norm = stack.enter_context(NativeNormRoPE(ordinal))
    sparse = stack.enter_context(NativeAttention(root / "native-attention-v1", ordinal, profile="long_context"))
    comp = stack.enter_context(NativeCompression(arrays, ordinal, layers=(2,)))
    index = stack.enter_context(NativeIndexer(arrays, ordinal, layers=(2,)))
    windows = {l: stack.enter_context(NativeWindowAttention(arrays, ps[l], norm, sparse, profile="token_prefix")) for l in (0, 1)}
    frame = stack.enter_context(SharedAttentionFrame(arrays, {l: ps[l] for l in (2, 3)}, norm, comp, index, sparse, (2, 3)))
    eg_arrays, eg_manifest = load_controls(root / "engram-fixture/controls1", 1)
    rows = stack.enter_context(EngramRows(root / "engram-fixture/rows/layer1", 1))
    engram = stack.enter_context(NativeEngram(eg_arrays, rows, ordinal))
    tokenizer = PinnedTokenizer(root / "import-source")
    mapping, mapping_identity = compressed_token_map(tokenizer)
    history = NgramState(mapping)
    sampler = stack.enter_context(NativeSampler(2026, ordinal))
    prefix = stack.enter_context(NativeTokenPrefix(arrays, embeddings, hc, norm, windows, frame, moe, engram, history, sampler))
    prefix.probe_operators = {"hc": hc, "projections": ps, "norm": norm, "sparse": sparse, "compressor": comp, "indexer": index}
    prefix.probe_identities = {"Engram1": eg_manifest, "rows1": rows.manifest, "tokenizer": mapping_identity}
    return prefix
