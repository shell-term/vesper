"""Spark-only request admission over the existing bounded native composition.

The default retains the historical 512-call profile. Opting into a larger
request does not enlarge shared operators or qualify their later positions.
"""

from dataclasses import dataclass
import math

from .generation import GenerationRequest
from .messages import MAX_PROMPT_CHARS
from .sequence import MAX_CALLS, scope as historical_scope
from .shared_attention import CAPACITY

MAX_CONTEXT_TOKENS = CAPACITY  # SharedAttentionFrame is the tightest state owner.
MAX_COMPLETION_CHARS = 16384  # Existing complete-response parser bound.
MAX_USER_PROMPT_CHARS = 4000
MAX_DURATION_SECONDS = 86400


@dataclass(frozen=True)
class SparkTextLimits:
    context_tokens: int = MAX_CALLS
    input_tokens: int = MAX_CALLS
    model_calls: int = MAX_CALLS
    duration_seconds: float | None = None

    def __post_init__(self):
        for name, value, maximum in (("context", self.context_tokens, MAX_CONTEXT_TOKENS),
                                     ("input", self.input_tokens, self.context_tokens),
                                     ("model-call", self.model_calls, self.context_tokens)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Expected an explicit {name} bound in [1,{maximum}]")
        seconds = self.duration_seconds
        if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds)
                                    or not 0 < seconds <= MAX_DURATION_SECONDS):
            raise ValueError("Expected a finite duration budget in (0,86400] seconds")
        if self.context_tokens > MAX_CALLS and seconds is None:
            raise ValueError("Extended Spark context requires an explicit finite duration budget")

    @property
    def metadata_multiplier(self):
        # Per-call scalar diagnostics grow with the admitted call count. The
        # model operators retain their existing bounded state/cache admissions.
        return (self.context_tokens + MAX_CALLS - 1) // MAX_CALLS

    def scope(self):
        value = historical_scope()
        if self.context_tokens != MAX_CALLS:
            value.update(profile="spark_bounded_serial_text_candidate",
                         maximum_model_calls=self.context_tokens,
                         maximum_input_embedding_rows=self.context_tokens,
                         maximum_Engram_rows_per_layer=24 * self.context_tokens)
        return {**value, "current_composition_position_capacity": MAX_CONTEXT_TOKENS,
                "extended_context_hardware_qualified": False,
                "maximum_serialized_input_characters": MAX_PROMPT_CHARS,
                "maximum_completion_characters": MAX_COMPLETION_CHARS}

    def admission(self, max_new_tokens):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.context_tokens:
            raise ValueError("Generated-token bound exceeds the explicit Spark context")
        return {"context_tokens": self.context_tokens, "input_tokens": self.input_tokens,
                "generated_tokens_including_eos": max_new_tokens, "model_calls": self.model_calls,
                "duration_seconds": self.duration_seconds,
                "duration_enforcement": "between_model_calls_and_preparation_phases; cleanup always runs",
                "one_inflight_model_call_may_overrun_duration": True,
                "maximum_completion_characters": MAX_COMPLETION_CHARS,
                "silent_truncation": False}


@dataclass(frozen=True)
class SparkTextRequest(GenerationRequest):
    limits: SparkTextLimits = SparkTextLimits()

    def encode(self, tokenizer):
        self.limits.admission(self.max_new_tokens)
        if (type(self.prompt) is not str or not self.prompt or len(self.prompt) > MAX_PROMPT_CHARS
                or type(self.eos_id) is not int or not 0 <= self.eos_id < 129280):
            raise ValueError("Expected the bounded original Spark text serialization and EOS")
        ids = tokenizer.encode(self.prompt)
        if (not ids or len(ids) > self.limits.input_tokens
                or any(type(i) is not int or not 0 <= i < 129280 for i in ids)):
            raise ValueError("Encoded input exceeds its explicit token bound or contains invalid IDs")
        if len(ids) + self.max_new_tokens - 1 > self.limits.context_tokens:
            raise ValueError("Input and continuation exceed the admitted Spark context")
        return tuple(ids)
