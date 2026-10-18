"""A bounded original layer-0 Transformer block, single rank and single token."""

from contextlib import ExitStack
import threading

from .connected_reference import native_connected_stages
from .hc import coefficients, hidden_shape
from .resources import MemoryBudget, MiB
from .window_attention import NativeWindowAttention


def validate_block_input(hidden, pre_mix, image_span):
    hidden_shape(hidden)
    if hidden.shape != (1, 1, 4, 5120) or type(image_span) is not bool:
        raise ValueError("Native layer-0 block currently supports one BF16 token and an explicit image-span flag")
    coefficients(pre_mix, (1, 1, 4))


class NativeBlock:
    """KV publication includes the MoE and final residual, with no route changes.

    Opened operators and original fixtures are supplied explicitly. Cache changes
    are permitted after failed calls; committed model state is restored or blocked.
    """

    def __init__(self, arrays, hc, normalization, projections, sparse_attention, moe):
        self.arrays, self.hc, self.norm, self.projections, self.sparse, self.moe = arrays, hc, normalization, projections, sparse_attention, moe
        self.attention = NativeWindowAttention(arrays, projections, normalization, sparse_attention)
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
            try:
                self.budget.acquire(32 * MiB)
                self.stack.callback(self.budget.release, 32 * MiB)
                self.stack.enter_context(self.attention)
                self.devices = [op.device for op in (self.hc, self.norm, self.projections, self.sparse)]
                self.devices += [self.attention.kv.device, *self.moe.devices.values()]
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
            return self.attention.recover()

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
            x = add("attn_norm_", self.norm.norm(x, self.arrays["layers.0.attn_norm.weight"]))
            def finish(attention_stages):
                add("attention_", attention_stages)
                after = add("attn_post_", self.hc.post(attention_stages["wo_b_output_bf16"], hidden, attn_mixes["post_fp32"], attn_mixes["comb_fp32"]))
                ffn_mixes = self.hc.mixes(after, "ffn")
                add("hc_ffn_", ffn_mixes)
                # The current FFN uses the preceding Attention pre coefficients.
                # Its own ffn_pre is returned for the next block's Attention.
                x = add("ffn_pre_", self.hc.pre(after, attn_mixes["pre_fp32"]))
                x = add("ffn_norm_", self.norm.norm(x, self.arrays["layers.0.ffn_norm.weight"]))
                self._headroom()
                result = self.moe.forward(x.reshape(5120), image_span)
                add("moe_", native_connected_stages(result))
                output = add("ffn_post_", self.hc.post(result["output_bf16"].reshape(1,1,5120), after, ffn_mixes["post_fp32"], ffn_mixes["comb_fp32"]))
                stages["output_bf16"] = output
                stages["next_pre_mix_fp32"] = ffn_mixes["pre_fp32"]
                self._headroom()
                return stages
            return self.attention.step(x, start_pos=start_pos, finalize=finish)

    def stats(self):
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "peak_owned_primary_device_bytes": self.owned_device_peak,
                "primary_device_cap_bytes": 512 * MiB, "reservation_is_RSS_limit": False}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()
