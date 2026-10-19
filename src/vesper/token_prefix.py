"""Actual embedding -> block0 -> Engram1 -> block1 -> blocks2/3 prefix.

Only this four-block diagnostic prefix is registered here. Its output is not a
40-layer model output and must not be offered as model text generation.
"""

import numpy as np

from .block_pair import BlockStages
from .token_transaction import TokenTransaction

LAYERS = (0, 1, 2, 3)
TOKENS = (0, 2)


class NativeTokenPrefix:
    def __init__(self, arrays, embeddings, hc, norm, windows, sharing, moe, engram, history, sampler):
        if (set(hc) != set(LAYERS) or set(moe) != set(LAYERS) or set(windows) != {0, 1}
                or sharing.layers != (2, 3) or engram.rows.layer != 1
                or any(hc[l].layer != l or moe[l].layer != l for l in LAYERS)
                or any(windows[l].layer != l for l in (0, 1))
                or not isinstance(embeddings, dict) or not set(TOKENS).issubset(embeddings)):
            raise ValueError("Expected the registered original 0..3 prefix bindings")
        for value in embeddings.values():
            if (not isinstance(value, np.ndarray) or value.shape != (5120,) or value.dtype != np.uint16
                    or not value.flags.c_contiguous or np.any((value & 0x7f80) == 0x7f80)):
                raise ValueError("Expected finite original BF16 embedding rows")
        self.arrays, self.embeddings, self.hc, self.norm = arrays, embeddings, hc, norm
        self.windows, self.sharing, self.moe, self.engram = windows, sharing, moe, engram
        self.transaction = TokenTransaction({"window0": windows[0], "window1": windows[1], "sharing2_3": sharing}, history, sampler)

    @property
    def position(self):
        return self.transaction.position

    def __enter__(self):
        operators = (*self.hc.values(), self.norm, *self.windows.values(), self.sharing, self.engram)
        if any(not o.opened for o in operators):
            raise RuntimeError("Token prefix requires open arithmetic/state operators")
        self.transaction.__enter__()
        return self

    def step(self, ids, *, start_pos=None, finalize=None):
        if (not isinstance(ids, np.ndarray) or ids.shape != (1, 1) or ids.dtype != np.int64
                or not ids.flags.c_contiguous or int(ids[0, 0]) not in self.embeddings
                or (finalize is not None and not callable(finalize))):
            raise ValueError("Expected one prepared original embedding token and optional diagnostic finalizer")
        def compute(candidate):
            hidden = np.repeat(self.embeddings[int(ids[0, 0])].reshape(1, 1, 1, 5120), 4, axis=2)
            # Unchanged make_identity_pre_mix selects stream zero, not a mean.
            pre = np.zeros((1, 1, 4), np.float32)
            pre[..., 0] = 1
            result = {"embedding_bf16": hidden.copy(), "initial_pre_fp32": pre.copy(),
                "hash_ids_i64": candidate.hash_ids.copy(), "compressed_i64": candidate.compressed.copy(),
                "lookbacks_i64": candidate.lookbacks.copy()}
            carry, phases = [hidden, pre], {}
            for layer in (0, 1):
                if layer == 1:
                    ids1 = np.ascontiguousarray(candidate.hash_ids[:, :, 0, :])
                    result["engram1"] = self.engram.forward(carry[0], ids1)
                    carry[0] = result["engram1"]["output_bf16"]
                phase = BlockStages(self.arrays, self.hc[layer], self.norm, layer, *carry)
                phases[layer] = phase
                def finish(attention, phase=phase, layer=layer):
                    x = phase.after_attention(attention["wo_b_output_bf16"])
                    output = self.moe[layer].forward(x.reshape(5120), False)
                    carry[:] = phase.finish(output)
                    phase.stages.update(("attention_" + n, a) for n, a in attention.items())
                self.windows[layer].step(phase.input, start_pos=candidate.start, finalize=finish)
            def prepare(layer):
                phases[layer] = BlockStages(self.arrays, self.hc[layer], self.norm, layer, *carry)
                return phases[layer].input
            def after(layer, attention):
                phase = phases[layer]
                x = phase.after_attention(attention["wo_b_output_bf16"])
                carry[:] = phase.finish(self.moe[layer].forward(x.reshape(5120), False))
            result["sharing_attention"] = self.sharing.step_connected(prepare, after, start_pos=candidate.start)
            result.update({f"layer{l}": phases[l].stages for l in LAYERS})
            result["output_bf16"], result["next_pre_mix_fp32"] = carry
            # Intended for bounded probe head/sampler/fault instrumentation, not
            # a substitute for the remaining 36 blocks.
            return result if finalize is None else finalize(result)
        return self.transaction.step(ids, compute, start_pos=start_pos)

    def recover(self):
        return self.transaction.recover()

    def __exit__(self, *error):
        return self.transaction.__exit__(*error)
