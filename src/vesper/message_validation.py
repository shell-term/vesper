"""An offline input/output format gate, explicitly separate from model quality."""

from contextlib import ExitStack
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

from .checkpoint import MODEL, REVISION, read_small
from .generation import MAX_CALLS, GenerationRequest
from .message_cases import encoding_cases, parsing_cases, rejection_cases, quality_cases
from .model_artifacts import FILES, canonical, digest, read_checked

SOURCE_FILES = {
    "encoding.py": (37962, "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"),
    "README.md": (12164, "a34a5ca98c65e5b5b2bedd693cc2e18e732fe50192723fb6937ae07219c23739"),
}
QUALIFIED_REFERENCE = "177d00d1b6ede9e415c5034c1018eaf60f8cba053a9cf37ae643196d0ade4bc2"


def corpus():
    return {"encoding": encoding_cases(), "parsing": parsing_cases(), "rejection": rejection_cases(), "quality_requests": quality_cases()}


def scope():
    return {"model": MODEL, "revision": REVISION, "corpus_sha256": canonical(corpus()),
        "whole_text_conversations": True, "original_BOS": True, "original_DSML_leading_spaces": True,
        "text_tools_and_results": True, "context_cache_run": False, "vision_run": False, "classification_tasks_run": False,
        "format_comparison_only": True, "model_inference_run": False, "model_quality_scored": False,
        "GPU_model_calls": 0, "model_weight_payload_bytes_fetched": 0, "execution_of_tools_or_generated_code": False}


def fetch_source(root):
    """Fetch only two small pinned format files; never replace existing bytes."""
    directory = Path(root) / "reference/encoding"; directory.mkdir(parents=True, exist_ok=True)
    for name, (size, sha) in SOURCE_FILES.items():
        path = directory / name
        if path.exists():
            if path.stat().st_size != size: raise ValueError("Pinned message source extent differs")
            raw = path.read_bytes()
        else:
            raw = read_small("encoding/" + name, limit=64 * 1024)
        if len(raw) != size or digest(raw) != sha: raise ValueError("Pinned message source differs")
        if not path.exists():
            with path.open("xb") as f: f.write(raw)
    # The upstream code's MIT license was imported with the pinned metadata.
    read_checked(Path(root) / "import-source", "LICENSE")
    return directory


def tokenizer_identity(root):
    return {name: digest(read_checked(Path(root) / "import-source", name)) for name in ("tokenizer.json", "tokenizer_config.json")}


def _capture(encoder, parser, tokenizer, *, original):
    encoded, parsed, rejected, requests = [], [], [], []
    for case in encoding_cases():
        before = canonical(case)
        prompt = encoder(case["messages"], **case["options"])
        ids = tokenizer.encode(prompt)
        if canonical(case) != before: raise ValueError("Message formatting mutated caller data")
        encoded.append({"id": case["id"], "prompt": prompt, "prompt_utf8_sha256": digest(prompt.encode()), "token_ids": ids})
    for case in parsing_cases():
        parsed.append({"id": case["id"], "parsed": parser(case["text"], case["thinking_mode"])})
    for case in rejection_cases():
        try:
            if case["operation"] == "encode": encoder(case["messages"], **case["options"])
            else: parser(case["text"], case["thinking_mode"])
        except (ValueError, AssertionError): rejected.append({"id": case["id"], "rejected": True})
        else: raise ValueError("A malformed format fixture was unexpectedly accepted: " + case["id"])
    for case in quality_cases():
        prompt = encoder(case["messages"], thinking_mode=case["thinking_mode"], reasoning_effort=case["reasoning_effort"])
        ids = tokenizer.encode(prompt); calls = len(ids) + case["max_new_tokens"] - 1
        fits = calls <= MAX_CALLS
        if not original:
            try: GenerationRequest(prompt, case["max_new_tokens"]).encode(tokenizer)
            except ValueError:
                if fits: raise ValueError("Message preflight and actual request admission disagree")
            else:
                if not fits: raise ValueError("Actual generation request exceeded its declared fixture scope")
        requests.append({"id": case["id"], "category": case["category"], "prompt": prompt,
            "prompt_utf8_sha256": digest(prompt.encode()), "prompt_token_ids": ids, "prompt_tokens": len(ids),
            "requested_max_new_tokens": case["max_new_tokens"], "maximum_model_calls_requested": calls,
            "current_fixture_maximum_calls": MAX_CALLS, "fits_current_fixture": fits,
            "status": "ready" if fits else "blocked_by_runtime_scope", "model_inference_run": False,
            "model_quality_scored": False, "completion": None, "score": None})
    return {"encoding": encoded, "parsing": parsed, "rejection": rejected, "quality_preflight": requests}


def capture_reference(root):
    """Run the entire unchanged encoder/parser and original AutoTokenizer on CPU."""
    directory = fetch_source(root)
    with ExitStack() as stack:
        http = stack.enter_context(patch("urllib.request.urlopen", side_effect=RuntimeError("Message comparison must remain offline")))
        # Execute exactly the checked source bytes, not a loader's potentially
        # stale timestamp-based .pyc. The entire original file is unchanged.
        path = directory / "encoding.py"
        size, sha = SOURCE_FILES[path.name]
        if path.stat().st_size != size: raise ValueError("Pinned executable message source extent differs")
        raw = path.read_bytes()
        if len(raw) != size or digest(raw) != sha: raise ValueError("Pinned executable message source differs")
        module = ModuleType("vesper_pinned_message_reference")
        exec(compile(raw, "<pinned-original-message-source>", "exec"), module.__dict__)
        from transformers import AutoTokenizer
        import torch
        before = torch.cuda.is_initialized()
        if before: raise ValueError("Expected a separate CPU-only message reference process")
        if importlib.metadata.version("transformers") != "5.19.0" or importlib.metadata.version("tokenizers") != "0.23.3":
            raise ValueError("Unqualified message reference tokenizer runtime")
        tokenizer = AutoTokenizer.from_pretrained(str(Path(root) / "import-source"), local_files_only=True, trust_remote_code=False)
        result = _capture(module.encode_messages, module.parse_message_from_completion_text, tokenizer, original=True)
        if torch.cuda.is_initialized(): raise ValueError("CPU format comparison initialized CUDA")
        report = {"schema_version": 1, "probe": "original_message_format_goldens", "status": "pass", "scope": scope(),
            "source_files": {n: sha for n, (_, sha) in SOURCE_FILES.items()}, "tokenizer": tokenizer_identity(root),
            "runtime": {"transformers": "5.19.0", "tokenizers": "0.23.3", "CUDA_initialized_before": before, "CUDA_initialized_after": False},
            "python_http_disabled_during_execution": True, "HTTP_calls_during_execution": http.call_count, **result}
        report["complete_goldens_sha256"] = canonical(result)
        return report


def validate_reference(report, *, require_qualified=True):
    if (report.get("probe"), report.get("status"), report.get("scope")) != ("original_message_format_goldens", "pass", scope()):
        raise ValueError("Different original message scope")
    if report["source_files"] != {n: sha for n, (_, sha) in SOURCE_FILES.items()}:
        raise ValueError("Different original message source")
    if report["tokenizer"] != {n: FILES[n][2] for n in ("tokenizer.json", "tokenizer_config.json")}:
        raise ValueError("Different original message tokenizer files")
    if report["runtime"] != {"transformers": "5.19.0", "tokenizers": "0.23.3", "CUDA_initialized_before": False, "CUDA_initialized_after": False}:
        raise ValueError("Different reference execution or tokenizer runtime")
    if report["python_http_disabled_during_execution"] is not True or report["HTTP_calls_during_execution"] != 0:
        raise ValueError("Original message execution was not offline")
    validate_results(report)
    if require_qualified and (not QUALIFIED_REFERENCE or canonical(report) != QUALIFIED_REFERENCE):
        raise ValueError("Original message capture is not independently qualified")
    return True


def validate_results(report):
    result = {n: report[n] for n in ("encoding", "parsing", "rejection", "quality_preflight")}
    if report["complete_goldens_sha256"] != canonical(result): raise ValueError("Changed complete message goldens")
    for name, cases in (("encoding", encoding_cases()), ("parsing", parsing_cases()), ("rejection", rejection_cases()), ("quality_preflight", quality_cases())):
        if [r["id"] for r in result[name]] != [c["id"] for c in cases]: raise ValueError("Missing or reordered format cases")
    for r in (*result["encoding"], *result["quality_preflight"]):
        if digest(r["prompt"].encode()) != r["prompt_utf8_sha256"]: raise ValueError("Changed original prompt bytes")
        ids = r.get("token_ids", r.get("prompt_token_ids"))
        if not ids or len(ids) > 4096 or any(type(i) is not int or not 0 <= i < 129280 for i in ids): raise ValueError("Invalid original prompt tokens")
    if any(r != {"id": c["id"], "rejected": True} for r, c in zip(result["rejection"], rejection_cases())):
        raise ValueError("A malformed format fixture was accepted")
    for r, c in zip(result["quality_preflight"], quality_cases()):
        if (r["category"], r["prompt_tokens"], r["requested_max_new_tokens"], r["maximum_model_calls_requested"], r["current_fixture_maximum_calls"],
                r["fits_current_fixture"], r["status"], r["model_inference_run"], r["model_quality_scored"], r["completion"], r["score"]) != (
                c["category"], len(r["prompt_token_ids"]), c["max_new_tokens"], len(r["prompt_token_ids"]) + c["max_new_tokens"] - 1, MAX_CALLS,
                False, "blocked_by_runtime_scope", False, False, None, None):
            raise ValueError("Unexecuted quality task was counted as a model evaluation")


def capture_native(root, original, original_raw):
    validate_reference(original)
    from .messages import encode_messages, parse_completion
    from .tokenizer import PinnedTokenizer
    frameworks = {n: {"installed": importlib.util.find_spec(n) is not None, "imported": n in sys.modules}
        for n in ("torch", "tilelang", "transformers", "sympy")}
    if any(any(info.values()) for info in frameworks.values()): raise ValueError("Expected independent framework-free native process")
    with patch("urllib.request.urlopen", side_effect=RuntimeError("Native message comparison must remain offline")) as http:
        result = _capture(encode_messages, lambda text, mode: parse_completion(text, thinking_mode=mode),
            PinnedTokenizer(Path(root) / "import-source"), original=False)
        tokenizer = tokenizer_identity(root)
    for name in result:
        if result[name] != original[name]: raise ValueError("Native message format differs: " + name)
    if tokenizer != original["tokenizer"]: raise ValueError("Different original tokenizer assets")
    report = {"schema_version": 1, "probe": "native_message_format_comparison", "status": "pass", "scope": scope(),
        "reference_sha256": digest(original_raw), "qualified_reference_sha256": QUALIFIED_REFERENCE, "tokenizer": tokenizer,
        "reference_frameworks": frameworks, "python_http_disabled_during_execution": True, "HTTP_calls_during_execution": http.call_count, **result}
    report["complete_goldens_sha256"] = canonical(result)
    return report


def validate_native(report, original, original_raw):
    validate_reference(original); validate_results(report)
    if (report.get("probe"), report.get("status"), report.get("scope"), report.get("reference_sha256"),
            report.get("qualified_reference_sha256"), report.get("tokenizer")) != (
            "native_message_format_comparison", "pass", scope(), digest(original_raw), QUALIFIED_REFERENCE, original["tokenizer"]):
        raise ValueError("Native message report is not bound to the actual sealed reference")
    if report["reference_frameworks"] != {n: {"installed": False, "imported": False} for n in ("torch", "tilelang", "transformers", "sympy")}:
        raise ValueError("Native message comparison imported a reference framework")
    if report["python_http_disabled_during_execution"] is not True or report["HTTP_calls_during_execution"] != 0:
        raise ValueError("Native message execution was not offline")
    for name in ("encoding", "parsing", "rejection", "quality_preflight"):
        if report[name] != original[name]: raise ValueError("Native message comparison omits a complete original result")
    return True


def read_report(path):
    """Bound before reading: reject accidentally selected checkpoints/reports."""
    path = Path(path); limit = 2 * 1024**2
    if not 1 <= path.stat().st_size <= limit: raise ValueError("Excessive message report metadata")
    with path.open("rb") as f: raw = f.read(limit + 1)
    if len(raw) > limit: raise ValueError("Message report grew beyond its metadata bound")
    return json.loads(raw), raw
