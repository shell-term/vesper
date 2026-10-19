"""Pinned backbone KV/key/index ownership; no inferred cache routing."""

from dataclasses import dataclass

KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
GROUPS = ((2, 3), (8, 9), (14, 15), (20, 21, 24, 25, 28, 29, 32, 33, 36, 37))
QUALIFIED_INPUT_LAYERS = tuple(l for group in GROUPS for l in group)


@dataclass(frozen=True)
class AttentionRole:
    layer: int
    ratio: int
    kv_owner: int | None
    index_owner: int | None
    candidate_owner: int | None

    @property
    def owns_kv(self): return self.layer == self.kv_owner

    @property
    def owns_index(self): return self.layer == self.index_owner


def role(layer):
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("Expected a pinned backbone Attention layer")
    if layer < 2: return AttentionRole(layer, 0, None, None, None)
    return AttentionRole(layer, 2 if layer < 20 else 1,
        max(l for l in KV_SOURCES if l <= layer), max(l for l in INDEX_SOURCES if l <= layer), 20 if layer >= 20 else None)


def validate_group(layers):
    if type(layers) is not tuple or layers not in GROUPS:
        raise ValueError("Unregistered bounded Attention sharing group")
    return layers


def compression_layout(layers):
    if (type(layers) is not tuple or not layers or len(set(layers)) != len(layers)
            or any(type(l) is not int or l not in KV_SOURCES for l in layers)):
        raise ValueError("Expected registered original compressor owners")
    return {f"layers.{l}.attn.compressor.{p}.weight": s for l in layers
            for p, s in (("wkv", (512, 5120)), ("norm", (512,))) + ((("wgate", (512, 5120)),) if role(l).ratio == 2 else ())}


def indexer_layout(layers):
    if (type(layers) is not tuple or not layers or len(set(layers)) != len(layers)
            or any(type(l) is not int or l not in INDEX_SOURCES for l in layers)):
        raise ValueError("Expected registered original index sources")
    rows = {}
    for l in layers:
        parts = {"wq_b.weight": ("F8_E4M3", (4096, 1280)), "wq_b.scale": ("F8_E8M0", (128, 40)),
                 "weights_proj.weight": ("BF16", (32, 5120))}
        if role(l).owns_kv: parts.update({"wk.weight": ("BF16", (128, 512)), "k_norm.weight": ("BF16", (128,))})
        rows.update((f"layers.{l}.attn.indexer.{p}", spec) for p, spec in parts.items())
    return rows


def validate_config(config):
    expected = {"n_layers": 40, "dim": 5120, "n_heads": 64, "head_dim": 512,
        "q_lora_rank": 1280, "o_groups": 8, "o_lora_rank": 1024, "window_size": 128,
        "kv_source_layers": list(KV_SOURCES), "index_source_layers": list(INDEX_SOURCES),
        "candidate_source_layer": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8}
    if any(config.get(k) != v for k, v in expected.items()) or config.get("compress_ratios", [])[:40] != [0, 0] + [2] * 18 + [1] * 20:
        raise ValueError("Pinned Attention topology/configuration differs")
    return [role(l) for l in range(40)]
