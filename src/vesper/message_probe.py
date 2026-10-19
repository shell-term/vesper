"""CPU-only reproduction of the pinned message format comparison.

This never runs model inference. ``reference`` needs the optional original
Transformers environment; ``native`` requires a framework-free environment.
Pinned tokenizer metadata must already be present under import-source.
"""

import argparse
import json
from pathlib import Path
import re

from .message_validation import capture_reference, capture_native, read_report, validate_reference, validate_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("reference", "native", "check"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--run", default="message-v1")
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run): p.error("Expected a simple message run name")
    run = args.artifacts / "message-format" / args.run
    original_path = args.reference or run / "original.json"
    if args.operation == "check":
        original, raw = read_report(original_path)
        native, _ = read_report(run / "native.json")
        validate_native(native, original, raw)
        print(json.dumps({"status": "pass", "format_comparison_only": True, "model_inference_run": False}))
        return
    run.mkdir(parents=True, exist_ok=True)
    target = run / ("original.json" if args.operation == "reference" else "native.json")
    if target.exists(): raise FileExistsError("Refusing to replace a completed message report")
    if args.operation == "reference":
        report = capture_reference(args.artifacts)
        validate_reference(report, require_qualified=False)
    else:
        original, raw = read_report(original_path)
        report = capture_native(args.artifacts, original, raw)
        validate_native(report, original, raw)
    with target.open("xb") as f: f.write((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"status": "pass", "probe": report["probe"], "format_comparison_only": True,
        "encoding_cases": len(report["encoding"]), "parser_cases": len(report["parsing"]), "rejection_cases": len(report["rejection"]),
        "quality_requests": [{k: r[k] for k in ("id", "prompt_tokens", "maximum_model_calls_requested", "status")} for r in report["quality_preflight"]]}))


if __name__ == "__main__": main()
