"""Explicit opt-in serial text scope; the historical eight-call probe is intact."""

from dataclasses import dataclass

from .generation import GenerationRequest

PROFILE = "bounded_serial_text_512"
MAX_CALLS, MAX_EMBEDDINGS, MAX_ROWS, MAX_EXPERTS_PER_LAYER = 512, 512, 12288, 384


def scope():
    return {"profile": PROFILE, "maximum_model_calls": MAX_CALLS, "maximum_input_embedding_rows": MAX_EMBEDDINGS,
        "maximum_Engram_rows_per_layer": MAX_ROWS, "maximum_original_experts_per_layer": MAX_EXPERTS_PER_LAYER,
        "queries_per_call": 1, "extra_quantization": False, "router_pruning_or_bias": False,
        "chunk_prefill_comparison_run": False, "full_model_quality": "not_run"}


@dataclass(frozen=True)
class SequenceRequest(GenerationRequest):
    """An explicitly larger request for the same feedback/EOS transaction driver.

    Admission alone does not prepare weights or qualify a model trajectory.
    The tokenizer keeps its existing 4,096-character bound.
    """
    def encode(self, tokenizer):
        if (type(self.prompt) is not str or not self.prompt or len(self.prompt) > 4096
                or type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= MAX_CALLS
                or type(self.eos_id) is not int or not 0 <= self.eos_id < 129280):
            raise ValueError("Expected an explicit bounded-512 text request")
        ids = tokenizer.encode(self.prompt)
        if (not ids or any(type(i) is not int or not 0 <= i < 129280 for i in ids)
                or len(ids) + self.max_new_tokens - 1 > MAX_CALLS):
            raise ValueError("Prompt and continuation exceed the bounded-512 serial scope")
        return tuple(ids)
