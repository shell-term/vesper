"""Bounded serial text continuation over an already prepared token engine.

Raw text is encoded without invented BOS/chat/thinking markers. This is not the
upstream message encoder or chunk-prefill implementation. The engine owns all
model state, original arithmetic and RNG; this driver only chooses the next ID.
"""

from dataclasses import dataclass
import numpy as np

MAX_CALLS = 8  # Explicit selected-expert fixture scope, not the model context.


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 3
    eos_id: int = 1

    def encode(self, tokenizer):
        if (type(self.prompt) is not str or not self.prompt or len(self.prompt) > 4096
                or type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= MAX_CALLS
                or type(self.eos_id) is not int or not 0 <= self.eos_id < 129280):
            raise ValueError("Expected a bounded explicit text generation request")
        ids = tokenizer.encode(self.prompt)
        if (not ids or any(type(i) is not int or not 0 <= i < 129280 for i in ids)
                or len(ids) + self.max_new_tokens - 1 > MAX_CALLS):
            raise ValueError("Prompt and continuation exceed the eight-call fixture scope")
        return tuple(ids)


class TextGeneration:
    """Feed successful samples back as input; a failed call publishes no ID.

    Retry is possible after the token engine restores its transaction. No saved
    reference trajectory, fallback token or weight fetch exists in this driver.
    EOS stops generation and is omitted from decoded completion, as upstream.
    """

    def __init__(self, tokenizer, step, request):
        if not callable(step): raise ValueError("Expected a prepared token engine")
        self.prompt_ids = request.encode(tokenizer)
        self.request, self.tokenizer, self.engine_step = request, tokenizer, step
        self.position, self.generated, self.stop_reason = 0, (), None

    @property
    def done(self): return self.stop_reason is not None

    def advance(self):
        if self.done: raise StopIteration("Text continuation is complete")
        pos = self.position
        token = self.prompt_ids[pos] if pos < len(self.prompt_ids) else self.generated[-1]
        sample = pos >= len(self.prompt_ids) - 1
        result = self.engine_step(np.array([[token]], np.int64), sample=sample)
        # The engine contract validates the returned selection before committing
        # its token transaction. Validate again for non-engine clients/tests.
        selected, reason, generated = None, None, self.generated
        if sample:
            ids = result["sampling"]["ids_i64"]
            if not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.shape != (1,) or not 0 <= int(ids[0]) < 129280:
                raise ValueError("Token engine returned an invalid original selection")
            selected = int(ids[0]); generated = (*generated, selected)
            if selected == self.request.eos_id: reason = "eos"
            elif len(generated) == self.request.max_new_tokens: reason = "max_new_tokens"
        self.position, self.generated, self.stop_reason = pos + 1, generated, reason
        return {"position": pos, "token_id": token, "sampling_requested": sample, "selected_id": selected}, result

    def summary(self):
        completion = list(self.generated)
        if completion and completion[-1] == self.request.eos_id: completion.pop()
        return {"prompt": self.request.prompt, "prompt_ids": list(self.prompt_ids), "max_new_tokens": self.request.max_new_tokens,
            "eos_id": self.request.eos_id, "calls": self.position, "selected_ids_including_eos": list(self.generated),
            "completion_ids": completion, "completion_text": self.tokenizer.decode(completion), "stop_reason": self.stop_reason}
