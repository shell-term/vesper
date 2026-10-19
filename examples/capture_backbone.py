"""Capture unchanged original Transformer/Block all-layer goldens first."""

import argparse
import json
from pathlib import Path
from vesper.backbone_reference import capture_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture_original(args.artifacts, lambda r: print(json.dumps(r), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__": main()
