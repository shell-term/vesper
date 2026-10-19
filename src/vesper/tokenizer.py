"""Original Rust tokenizer and compressed vocabulary, without Transformers."""

import importlib.metadata
import numpy as np

from .model_artifacts import read_checked, digest
from .resources import MemoryBudget, MiB

TOKENIZERS_VERSION = "0.23.3"
COMPRESSED_MAP_SHA256 = "26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496"


class PinnedTokenizer:
    def __init__(self, root):
        if importlib.metadata.version("tokenizers") != TOKENIZERS_VERSION: raise ValueError("Unqualified tokenizer runtime version")
        from tokenizers import Tokenizer
        read_checked(root,"tokenizer_config.json")
        raw = read_checked(root,"tokenizer.json")
        self.backend_tokenizer = Tokenizer.from_str(raw.decode())
        self.size = self.backend_tokenizer.get_vocab_size(with_added_tokens=True)
        if self.size != 129280: raise ValueError("Original tokenizer vocabulary differs")

    def __len__(self): return self.size

    def encode(self, text):
        if type(text) is not str or len(text) > 4096: raise ValueError("Expected a bounded text prompt")
        return self.backend_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        if len(ids)>4096 or any(type(i) is not int or not 0<=i<self.size for i in ids): raise ValueError("Invalid tokenizer IDs")
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)


def compressed_token_map(tokenizer, *, require_qualified=True):
    """Adapted from the pinned MIT Engram source; use the same Rust normalizers."""
    from tokenizers import Regex, normalizers
    normalizer = normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+")," "),normalizers.Replace(Regex(r"^ $"),"\ue000"),normalizers.Strip(),normalizers.Replace("\ue000"," ")])
    budget=MemoryBudget(cap=128*MiB); budget.acquire(budget.cap)
    try:
        keys, result = {}, np.empty(len(tokenizer),np.int64)
        for i in range(len(tokenizer)):
            text=tokenizer.backend_tokenizer.decode([i],skip_special_tokens=False)
            key=tokenizer.backend_tokenizer.id_to_token(i) if "\ufffd" in text else normalizer.normalize_str(text) or text
            if key not in keys: keys[key]=len(keys)
            result[i]=keys[key]
        sha=digest(result.tobytes())
        if len(keys)!=99092 or (require_qualified and (not COMPRESSED_MAP_SHA256 or sha!=COMPRESSED_MAP_SHA256)):
            raise ValueError("Unqualified original compressed tokenizer vocabulary")
        result.flags.writeable=False
        return result,{"original_vocab_size":len(tokenizer),"compressed_vocab_size":len(keys),"compressed_map_sha256":sha,
            "tokenizers_version":TOKENIZERS_VERSION,"host_allowance_bytes":budget.peak,"reservation_is_RSS_limit":False}
    finally: budget.release(budget.cap)
