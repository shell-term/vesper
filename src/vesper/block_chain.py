"""Complete consecutive 20..25 blocks with real index refresh and atomic state."""

from .attention_topology import REFRESH_CHAIN, validate_block_chain
from .block_pair import NativeBlockPair
from .resources import GiB, MiB


class NativeBlockChain(NativeBlockPair):
    _validate_layers = staticmethod(validate_block_chain)
    primary_cap, helper_cap, host_cap = 2 * GiB, 384 * MiB, 96 * MiB

    def __init__(self, arrays, hc, projections, norm, compressor, indexer, attention, moe, *, layers=REFRESH_CHAIN):
        super().__init__(arrays, hc, projections, norm, compressor, indexer, attention, moe, layers=layers)

    def _result(self, phases, attention, carry):
        result = {f"layer{l}": phases[l].stages for l in self.layers}
        result.update(attention=attention,
            handoffs={l: {"hidden_bf16": phases[l].hidden.copy(),
                          "pre_mix_fp32": phases[l].stages["incoming_pre_fp32"].copy()} for l in self.layers[1:]},
            output_bf16=carry[0], next_pre_mix_fp32=carry[1])
        return result
