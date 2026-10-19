"""Bounded original layer-0/2 Transformer blocks, single rank and single token."""

from contextlib import ExitStack
import threading

from .connected_reference import native_connected_stages
from .hc import coefficients, hidden_shape
from .resources import MemoryBudget, MiB
from .shared_attention import SharedAttentionFrame
from .window_attention import NativeWindowAttention


def validate_block_input(hidden, pre_mix, image_span):
    hidden_shape(hidden)
    if hidden.shape != (1, 1, 4, 5120) or type(image_span) is not bool:
        raise ValueError("Native block currently supports one BF16 token and an explicit image-span flag")
    coefficients(pre_mix, (1, 1, 4))


class NativeBlock:
    """KV publication includes the MoE and final residual, with no route changes.

    Opened operators and original fixtures are supplied explicitly. Cache changes
    are permitted after failed calls; committed model state is restored or blocked.
    """

    def __init__(self, arrays, hc, normalization, projections, sparse_attention, moe, *, layer=0, compressor=None, indexer=None):
        if type(layer) is not int or layer not in (0, 2):
            raise ValueError("Unregistered complete Transformer block layer")
        if (layer == 0 and (compressor is not None or indexer is not None)) or (layer == 2 and (compressor is None or indexer is None)):
            raise ValueError("Block compression/index operators differ from the original layer role")
        self.layer, self.compressor, self.indexer = layer, compressor, indexer
        self.arrays, self.hc, self.norm, self.projections, self.sparse, self.moe = arrays, hc, normalization, projections, sparse_attention, moe
        self.attention = (NativeWindowAttention(arrays, projections, normalization, sparse_attention) if layer == 0 else
            SharedAttentionFrame(arrays, {layer: projections}, normalization, compressor, indexer, sparse_attention, (layer,)))
        self.stack, self.lock, self.budget = ExitStack(), threading.Lock(), MemoryBudget(cap=32 * MiB)
        self.opened = False
        self.owned_device_peak = 0
        self.allocation_lock = threading.RLock()

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Native block is already open")
            if not self.hc.opened or self.hc.ordinal != self.projections.ordinal or self.moe.stack is None:
                raise ValueError("Block operators must be open on the same qualified device")
            if self.moe.config.cache.primary_device != self.projections.ordinal:
                raise ValueError("MoE and Attention compute devices differ")
            if any(getattr(op, "layer", None) != self.layer for op in (self.hc, self.projections, self.moe)):
                raise ValueError("Block HC, projection and MoE layer identities differ")
            try:
                self.budget.acquire(32 * MiB)
                self.stack.callback(self.budget.release, 32 * MiB)
                self.stack.enter_context(self.attention)
                self.devices = [op.device for op in (self.hc, self.norm, self.projections, self.sparse)]
                self.devices += ([self.attention.kv.device] if self.layer == 0 else
                    [self.compressor.device, self.compressor.norm.device, self.indexer.device, self.indexer.norm.device])
                self.devices += list(self.moe.devices.values())
                if len({id(d) for d in self.devices}) != len(self.devices):
                    raise ValueError("Block device ledgers must have distinct ownership")
                for device in self.devices:
                    original = device.allocate
                    def allocate(size, original=original):
                        with self.allocation_lock:
                            if self._owned_total() + size > 512 * MiB:
                                raise MemoryError("Block allocation exceeds the combined owned GPU cap")
                            pointer = original(size)
                            self.owned_device_peak = max(self.owned_device_peak, self._owned_total())
                            return pointer
                    device.allocate = allocate
                    self.stack.callback(setattr, device, "allocate", original)
                self.opened = True
                self._headroom()
                return self
            except BaseException:
                self.opened = False
                self.stack.close()
                raise

    def _owned_total(self):
        return sum(sum(d.allocations.values()) for d in self.devices)

    def _headroom(self):
        self.budget.acquire(0)
        owned = self._owned_total()
        self.owned_device_peak = max(self.owned_device_peak, owned)
        if owned > 512 * MiB:
            raise MemoryError("Native block owned GPU buffers exceed the scoped 512-MiB cap")
        if any(d.memory()["free"] < d.free_floor for d in self.devices):
            raise MemoryError("Native block refuses work below a component VRAM reserve")

    @property
    def position(self):
        return self.attention.position

    def reset(self):
        with self.lock:
            if not self.opened:
                raise RuntimeError("Native block is not open")
            self.attention.reset()

    def recover(self):
        with self.lock:
            if not self.opened:
                raise RuntimeError("Native block is not open")
            if self.layer == 0:
                return self.attention.recover()
            # Host candidate frames never publish partial model state.
            self.attention._ready()
            return self.attention.inspect()

    def step(self, hidden, pre_mix, image_span=False, *, start_pos=None):
        with self.lock:
            if not self.opened:
                raise RuntimeError("Native block is not open")
            self.attention._ready()
            validate_block_input(hidden, pre_mix, image_span)
            if start_pos is not None and (type(start_pos) is not int or start_pos != self.position):
                raise ValueError("Native block requires the next exact committed position")
            self._headroom()
            stages = {}
            def add(prefix, values):
                stages.update((prefix + name, a) for name,a in values.items())
                return values.get("output_bf16")
            attn_mixes = self.hc.mixes(hidden, "attn")
            add("hc_attn_", attn_mixes)
            x = add("attn_pre_", self.hc.pre(hidden, pre_mix))
            x = add("attn_norm_", self.norm.norm(x, self.arrays[f"layers.{self.layer}.attn_norm.weight"]))
            def finish(attention_stages):
                add("attention_", attention_stages)
                output_key = "wo_b_output_bf16" if self.layer == 0 else f"layer{self.layer}_wo_b_output_bf16"
                after = add("attn_post_", self.hc.post(attention_stages[output_key], hidden, attn_mixes["post_fp32"], attn_mixes["comb_fp32"]))
                ffn_mixes = self.hc.mixes(after, "ffn")
                add("hc_ffn_", ffn_mixes)
                # The current FFN uses the preceding Attention pre coefficients.
                # Its own ffn_pre is returned for the next block's Attention.
                x = add("ffn_pre_", self.hc.pre(after, attn_mixes["pre_fp32"]))
                x = add("ffn_norm_", self.norm.norm(x, self.arrays[f"layers.{self.layer}.ffn_norm.weight"]))
                self._headroom()
                result = self.moe.forward(x.reshape(5120), image_span)
                add("moe_", native_connected_stages(result))
                output = add("ffn_post_", self.hc.post(result["output_bf16"].reshape(1,1,5120), after, ffn_mixes["post_fp32"], ffn_mixes["comb_fp32"]))
                stages["output_bf16"] = output
                stages["next_pre_mix_fp32"] = ffn_mixes["pre_fp32"]
                self._headroom()
                return stages
            values = x if self.layer == 0 else {self.layer: x}
            return self.attention.step(values, start_pos=start_pos, finalize=finish)

    def stats(self):
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "peak_owned_primary_device_bytes": self.owned_device_peak,
                "primary_device_cap_bytes": 512 * MiB, "reservation_is_RSS_limit": False}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()
