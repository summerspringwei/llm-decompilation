"""Trace named internal ELF functions with GDB breakpoints.

This works for binaries retaining a symbol table. For fully stripped binaries,
the caller should first map Ghidra entries to runtime addresses (ASLR-aware) and
extend the breakpoint generation step accordingly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace internal calls made by an ELF invocation.")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--function", action="append", required=True, help="Symbol to trace; repeatable.")
    parser.add_argument("--arg", action="append", default=[], help="Argument passed to the binary; repeatable.")
    parser.add_argument("--stdin", help="File used as the inferior's standard input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = ["set pagination off", "set confirm off", "set breakpoint pending on", "set disable-randomization on"]
    for function in args.function:
        lines.extend([
            "break " + function,
            "commands",
            "silent",
            'printf "@@TRACE@@ ' + function + '\\n"',
            "continue",
            "end",
        ])
    if args.stdin:
        lines.extend(["run < " + str(Path(args.stdin).resolve()), "quit"])
    else:
        lines.extend(["run", "quit"])
    with tempfile.NamedTemporaryFile("w", suffix=".gdb", delete=False) as script:
        script.write("\n".join(lines) + "\n")
        script_path = script.name
    try:
        completed = subprocess.run(["gdb", "--batch", "-x", script_path, "--args", args.binary, *args.arg], text=True, capture_output=True, timeout=300)
    finally:
        Path(script_path).unlink(missing_ok=True)
    trace = re.findall(r"^@@TRACE@@ (.+)$", completed.stdout, flags=re.MULTILINE)
    output = {
        "binary": str(Path(args.binary).resolve()),
        "arguments": args.arg,
        "stdin": args.stdin,
        "requested_functions": args.function,
        "hits": trace,
        "unique_hits": sorted(set(trace)),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print("hits: " + ", ".join(output["unique_hits"]))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
