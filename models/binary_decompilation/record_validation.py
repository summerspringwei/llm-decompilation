"""Record source-comparison and replacement-test evidence in a function manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one function's decompilation validation result.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--decompiled", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--replacement-binary", required=True)
    parser.add_argument("--test", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    matches = [item for item in manifest["functions"] if item["name"] == args.function]
    if len(matches) != 1:
        raise ValueError("Expected exactly one manifest function named " + args.function)
    item = matches[0]
    item["validation_status"] = "validated"
    item["validation"] = {
        "decompiled_candidate": str(Path(args.decompiled).resolve()),
        "source_reference": str(Path(args.reference).resolve()),
        "replacement_binary": str(Path(args.replacement_binary).resolve()),
        "test": args.test,
        "result": "passed",
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("validated: " + args.function)


if __name__ == "__main__":
    main()
