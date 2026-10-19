"""Registered complete block pairs with actual HC handoff and atomic sharing."""

from contextlib import ExitStack
import threading

from .block import validate_block_input
from .attention_topology import validate_block_pair
from .connected_reference import native_connected_stages
from .resources import MemoryBudget, MiB
from .shared_attention import SharedAttentionFrame

LAYERS = (2, 3)
PRIMARY_CAP, HELPER_CAP = 768 * MiB, 128 * MiB


class BlockStages:
    """Stateless native block phases, also usable by the explicit prefix planner."""
    def __init__(self, arrays, hc, norm, layer, hidden, pre):
        self.arrays, self.hc, self.norm, self.layer, self.hidden = arrays, hc, norm, layer, hidden
        self.stages = {"input_hidden_bf16": hidden.copy(), "incoming_pre_fp32": pre.copy()}
        self.attn_mix = hc.mixes(hidden, "attn")
        self.add("hc_attn_", self.attn_mix)
        x = self.add("attn_pre_", hc.pre(hidden, pre))
        self.input = self.add("attn_norm_", norm.norm(x, arrays[f"layers.{layer}.attn_norm.weight"]))

    def add(self, prefix, values):
        self.stages.update((prefix + n, a) for n, a in values.items())
        return values.get("output_bf16")

    def after_attention(self, output):
        self.after = self.add("attn_post_", self.hc.post(output, self.hidden, self.attn_mix["post_fp32"], self.attn_mix["comb_fp32"]))
        self.ffn_mix = self.hc.mixes(self.after, "ffn")
        self.add("hc_ffn_", self.ffn_mix)
        x = self.add("ffn_pre_", self.hc.pre(self.after, self.attn_mix["pre_fp32"]))
        return self.add("ffn_norm_", self.norm.norm(x, self.arrays[f"layers.{self.layer}.ffn_norm.weight"]))

    def finish(self, moe):
        self.add("moe_", native_connected_stages(moe))
        output = self.add("ffn_post_", self.hc.post(moe["output_bf16"].reshape(1, 1, 5120), self.after,
            self.ffn_mix["post_fp32"], self.ffn_mix["comb_fp32"]))
        self.stages["output_bf16"], self.stages["next_pre_mix_fp32"] = output, self.ffn_mix["pre_fp32"]
        return output, self.ffn_mix["pre_fp32"]


class NativeBlockPair:
    def __init__(self, arrays, hc, projections, norm, compressor, indexer, attention, moe, *, layers=LAYERS):
        self.layers = validate_block_pair(layers)
        if any(not isinstance(operators, dict) or set(operators) != set(layers) for operators in (hc, projections, moe)):
            raise ValueError("Complete pair requires exactly its original layer operators")
        self.arrays, self.hc, self.ps, self.norm, self.comp, self.index, self.sparse, self.moe = arrays, hc, projections, norm, compressor, indexer, attention, moe
        self.attention = SharedAttentionFrame(arrays, projections, norm, compressor, indexer, attention, layers)
        self.stack, self.lock, self.allocation_lock = ExitStack(), threading.Lock(), threading.RLock()
        self.budget, self.opened, self.primary_peak, self.helper_peak = MemoryBudget(cap=64 * MiB), False, 0, 0

    def __enter__(self):
        with self.lock:
            if self.opened: raise RuntimeError("Complete block pair is already open")
            ordinal = self.ps[self.layers[0]].ordinal
            if (any(not self.hc[l].opened or self.hc[l].ordinal != ordinal or self.hc[l].layer != l or self.ps[l].layer != l
                    or self.moe[l].stack is None or self.moe[l].layer != l or self.moe[l].config.cache.primary_device != ordinal for l in self.layers)):
                raise ValueError("Pair HC, projection and MoE original identities/devices differ")
            try:
                self.budget.acquire(64 * MiB); self.stack.callback(self.budget.release, 64 * MiB)
                self.stack.enter_context(self.attention)
                self.devices = [o.device for o in (*self.hc.values(), *self.ps.values(), self.norm, self.comp, self.comp.norm, self.index, self.index.norm, self.sparse)]
                self.devices += [d for m in self.moe.values() for d in m.devices.values()]
                self.helpers = [m.helper_device for m in self.moe.values() if m.helper_device is not None]
                if len({id(d) for d in self.devices + self.helpers}) != len(self.devices + self.helpers):
                    raise ValueError("Pair device ledgers must have distinct ownership")
                for devices, cap in ((self.devices, PRIMARY_CAP), (self.helpers, HELPER_CAP)):
                    for device in devices:
                        original = device.allocate
                        def allocate(size, original=original, devices=devices, cap=cap):
                            with self.allocation_lock:
                                if self._total(devices) + size > cap: raise MemoryError("Combined block-pair GPU allocation cap exceeded")
                                pointer = original(size)
                                self._peaks()
                                return pointer
                        device.allocate = allocate
                        self.stack.callback(setattr, device, "allocate", original)
                self.opened = True
                self._headroom()
                return self
            except BaseException:
                self.opened = False; self.stack.close()
                raise

    @staticmethod
    def _total(devices): return sum(sum(d.allocations.values()) for d in devices)

    def _peaks(self):
        self.primary_peak = max(self.primary_peak, self._total(self.devices))
        self.helper_peak = max(self.helper_peak, self._total(self.helpers))

    def _headroom(self):
        self.budget.acquire(0)
        self._peaks()
        if self._total(self.devices) > PRIMARY_CAP or self._total(self.helpers) > HELPER_CAP:
            raise MemoryError("Complete pair exceeds its combined GPU caps")
        # memory() selects its device. Finish on the compute GPU before callers
        # enter a reference framework or load another SM89 kernel.
        if any(d.memory()["free"] < d.free_floor for d in self.helpers + self.devices):
            raise MemoryError("Complete pair refuses work below its GPU reserve")

    @property
    def position(self): return self.attention.position

    def reset(self):
        with self.lock:
            if not self.opened: raise RuntimeError("Complete pair is not open")
            self.attention.reset()

    def step(self, hidden, pre_mix, image_span=False, *, start_pos=None):
        with self.lock:
            if not self.opened: raise RuntimeError("Complete pair is not open")
            validate_block_input(hidden, pre_mix, image_span)
            if start_pos is not None and (type(start_pos) is not int or start_pos != self.position):
                raise ValueError("Complete pair requires its next committed position")
            self._headroom()
            phases, carry = {}, [hidden, pre_mix]
            def prepare(layer):
                self._headroom()
                phases[layer] = BlockStages(self.arrays, self.hc[layer], self.norm, layer, *carry)
                return phases[layer].input
            def finish(layer, attention):
                x = phases[layer].after_attention(attention["wo_b_output_bf16"])
                self._headroom()
                result = self.moe[layer].forward(x.reshape(5120), image_span)
                carry[:] = phases[layer].finish(result)
                self._headroom()
            def finalize(attention):
                result = {f"layer{l}": phases[l].stages for l in self.layers}
                consumer = phases[self.layers[1]]
                result.update(attention=attention, handoff_hidden_bf16=consumer.hidden.copy(),
                    handoff_pre_mix_fp32=consumer.stages["incoming_pre_fp32"].copy(), output_bf16=carry[0], next_pre_mix_fp32=carry[1])
                self._headroom()
                return result
            return self.attention.step_connected(prepare, finish, start_pos=start_pos, finalize=finalize)

    def stats(self):
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
            "primary_cap_bytes": PRIMARY_CAP, "helper_cap_bytes": HELPER_CAP,
            "peak_owned_primary_bytes": self.primary_peak, "peak_owned_helper_bytes": self.helper_peak,
            "reservation_is_RSS_limit": False}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False; self.stack.close()
