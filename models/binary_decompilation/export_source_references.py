"""Extract reference C definitions for functions listed in a binary manifest.

References are verification inputs. This utility never feeds them into the
binary decompiler or generated candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export source references for decompiled binary functions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def skip_string_or_comment(text: str, index: int) -> int:
    if text.startswith("//", index):
        end = text.find("\n", index)
        return len(text) if end < 0 else end + 1
    if text.startswith("/*", index):
        end = text.find("*/", index + 2)
        return len(text) if end < 0 else end + 2
    quote = text[index]
    if quote in "\"'":
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
            elif text[index] == quote:
                return index + 1
            else:
                index += 1
    return index


def balanced_end(text: str, index: int, opening: str, closing: str) -> int | None:
    depth = 0
    while index < len(text):
        if text.startswith("//", index) or text.startswith("/*", index) or text[index] in "\"'":
            index = skip_string_or_comment(text, index)
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def find_definition(text: str, name: str) -> str | None:
    span = find_definition_span(text, name)
    return text[span[0]:span[1]] if span is not None else None


def find_definition_span(text: str, name: str) -> tuple[int, int] | None:
    for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
        parameter_open = text.find("(", match.start(), match.end())
        parameter_close = balanced_end(text, parameter_open, "(", ")")
        if parameter_close is None:
            continue
        cursor = parameter_close + 1
        while cursor < len(text):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            # Some gnulib replacements select the public symbol with
            # preprocessor directives between the declarator and body.
            # Keep those directives inside the replacement span.
            line_start = text.rfind("\n", 0, cursor) + 1
            if cursor < len(text) and text[cursor] == "#" and cursor == line_start:
                line_end = text.find("\n", cursor)
                cursor = len(text) if line_end < 0 else line_end + 1
                continue
            break
        if cursor >= len(text) or text[cursor] != "{":
            continue
        body_end = balanced_end(text, cursor, "{", "}")
        if body_end is None:
            continue
        # Include contiguous declaration lines (e.g. ``static void``), but
        # stop at a preceding comment rather than replacing comment text.
        start = text.rfind("\n", 0, match.start()) + 1
        while start > 0:
            previous_end = start - 1
            previous_start = text.rfind("\n", 0, previous_end) + 1
            previous = text[previous_start:previous_end].strip()
            if (not previous or previous == "}" or previous.startswith("}")
                    or previous.startswith("#") or previous.endswith(";")
                    or "/*" in previous or "*/" in previous or previous.startswith("//")):
                break
            start = previous_start
        return start, body_end + 1
    return None


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for function in manifest["functions"]:
        if function["runtime_scaffolding"]:
            continue
        reference = None
        source_path = None
        for candidate in function["source_candidates"]:
            extracted = find_definition(Path(candidate).read_text(), function["name"])
            if extracted is not None:
                reference, source_path = extracted, candidate
                break
        if reference is not None:
            path = output_dir / (function["name"] + ".reference.c")
            path.write_text(reference + "\n")
            results[function["entry"]] = {"source": source_path, "reference": str(path), "found": True}
        else:
            results[function["entry"]] = {"source": function["source_candidates"], "reference": "", "found": False}
    (output_dir / "reference_index.json").write_text(json.dumps(results, indent=2) + "\n")
    print("references_found: %d" % sum(item["found"] for item in results.values()))
    print("references_missing: %d" % sum(not item["found"] for item in results.values()))


if __name__ == "__main__":
    main()
