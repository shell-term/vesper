"""Explicit forty-layer native probe with both Engrams and original output tail."""

import numpy as np

from .attention_topology import BACKBONE_COMPRESSED_LAYERS
from .block_pair import BlockStages
from .token_transaction import TokenTransaction

LAYERS = tuple(range(40))


class NativeBackbone:
    def __init__(self, arrays, embeddings, hc, norm, windows, sharing, moe, engrams, history, sampler, head, norm_weight, *, temperature=.7):
        if (set(hc) != set(LAYERS) or set(moe) != set(LAYERS) or set(windows) != {0, 1}
                or set(engrams) != {1, 14} or sharing.layers != BACKBONE_COMPRESSED_LAYERS
                or sharing.binding_profile != "backbone_probe"
                or any(hc[l].layer != l or moe[l].layer != l for l in LAYERS)
                or any(windows[l].layer != l for l in (0, 1))
                or any(engrams[l].rows.layer != l for l in engrams)):
            raise ValueError("Expected every original layer and both Engram bindings")
        for token, value in embeddings.items():
            if (type(token) is not int or not 0 <= token < 129280 or value.shape != (5120,) or value.dtype != np.uint16
                    or not value.flags.c_contiguous or np.any((value & 0x7f80) == 0x7f80)):
                raise ValueError("Expected prepared finite original embedding rows")
        self.arrays, self.embeddings, self.hc, self.norm = arrays, embeddings, hc, norm
        self.windows, self.sharing, self.moe, self.engrams = windows, sharing, moe, engrams
        self.head, self.norm_weight, self.temperature = head, norm_weight, temperature
        self.transaction = TokenTransaction({"window0": windows[0], "window1": windows[1], "sharing2_39": sharing}, history, sampler)

    @property
    def position(self): return self.transaction.position

    def __enter__(self):
        if any(not op.opened for op in (*self.hc.values(), self.norm, *self.windows.values(), self.sharing, *self.engrams.values(), self.head)):
            raise RuntimeError("Backbone probe requires open original arithmetic/state operators")
        self.transaction.__enter__()
        return self

    def _evaluate(self, candidate, ids, resolve=None):
        """Stage actual layers. Only the artifact preparer supplies a resolver.

        The public step never supplies one and cannot fetch missing weights.
        Preparation can resolve an original route while the token is staged,
        avoiding quadratic replay of every preceding layer during import.
        """
        if candidate.compressed.shape[1] != 1: raise ValueError("Backbone probe is single-query only")
        hidden = np.repeat(self.embeddings[int(ids[0, 0])].reshape(1, 1, 1, 5120), 4, axis=2)
        pre = np.zeros((1, 1, 4), np.float32); pre[..., 0] = 1
        result = {"embedding_bf16": hidden.copy(), "initial_pre_fp32": pre.copy(), "hash_ids_i64": candidate.hash_ids.copy(),
            "compressed_i64": candidate.compressed.copy(), "lookbacks_i64": candidate.lookbacks.copy()}
        carry, phases = [hidden, pre], {}
        def prepare(layer):
            if layer in self.engrams:
                hashes = np.ascontiguousarray(candidate.hash_ids[:, :, (1, 14).index(layer), :])
                result[f"engram{layer}"] = self.engrams[layer].forward(carry[0], hashes)
                carry[0] = result[f"engram{layer}"]["output_bf16"]
            phase = BlockStages(self.arrays, self.hc[layer], self.norm, layer, *carry)
            phases[layer] = phase
            return phase.input
        def after(layer, attention):
            phase = phases[layer]; x = phase.after_attention(attention["wo_b_output_bf16"])
            values = x.reshape(5120)
            output = self.moe[layer].forward(values, False) if resolve is None else resolve(layer, values)
            carry[:] = phase.finish(output)
            if layer < 2: phase.stages.update(("attention_" + n, a) for n, a in attention.items())
        for layer in (0, 1):
            x = prepare(layer)
            self.windows[layer].step(x, start_pos=candidate.start, finalize=lambda attention, l=layer: after(l, attention))
        result["sharing_attention"] = self.sharing.step_connected(prepare, after, start_pos=candidate.start)
        result.update({f"layer{l}": phases[l].stages for l in LAYERS})
        result["output_bf16"], result["next_pre_mix_fp32"] = carry
        return result

    def _finish(self, result):
        collapsed = self.hc[39].pre(result["output_bf16"], result["next_pre_mix_fp32"])
        normalized = self.norm.norm(collapsed["output_bf16"], self.norm_weight)
        result["head"] = {**{"pre_" + n: a for n, a in collapsed.items()}, **{"norm_" + n: a for n, a in normalized.items()},
            **self.head.project(normalized["output_bf16"][:, -1].copy())}
        result["sampling"] = self.transaction.sampler.sample(result["head"]["logits_fp32"], self.temperature)
        return result

    def _validate_ids(self, ids):
        if (not isinstance(ids, np.ndarray) or ids.shape != (1, 1) or ids.dtype != np.int64 or not ids.flags.c_contiguous
                or int(ids[0, 0]) not in self.embeddings):
            raise ValueError("Expected one explicitly prepared original token embedding")

    def step(self, ids, *, start_pos=None, observe=None):
        self._validate_ids(ids)
        ids = ids.copy()
        if observe is not None and not callable(observe): raise ValueError("Expected an optional observer")
        def compute(candidate):
            result = self._finish(self._evaluate(candidate, ids))
            if observe is not None: observe(result)
            return result
        return self.transaction.step(ids, compute, start_pos=start_pos)

    def recover(self): return self.transaction.recover()

    def __exit__(self, *error): return self.transaction.__exit__(*error)
