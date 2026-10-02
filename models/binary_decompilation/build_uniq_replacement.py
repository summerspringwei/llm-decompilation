"""Build a Coreutils variant with recovered functions substituted.

The candidate is compiled in the original ``uniq.c`` translation unit, then
linked against the untouched original coreutils support libraries. This makes a
per-function test failure attributable to the replacement rather than to a
reconstructed dependency set.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from models.binary_decompilation.export_source_references import find_definition_span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Coreutils executable with recovered function replacements.")
    parser.add_argument("--function", action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[], help="Complete source-level replacement definition.")
    parser.add_argument(
        "--source-file",
        default="src/uniq.c",
        help="Path relative to --source-dir containing the target definition (default: src/uniq.c).",
    )
    parser.add_argument(
        "--original-object",
        action="append",
        default=[],
        help=(
            "Object path relative to --build-dir to link unchanged, repeatable. "
            "Use this when --source-file does not define main."
        ),
    )
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--coreutils-library",
        default="lib/libcoreutils.a",
        help="Coreutils support archive relative to --build-dir (default: lib/libcoreutils.a).",
    )
    parser.add_argument("--clang", default="clang")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / args.source_file
    if not args.function or len(args.function) != len(args.candidate):
        raise ValueError("Provide matching, repeatable --function and --candidate options")

    original = source.read_text()
    replacements = []
    for function, candidate_path in zip(args.function, args.candidate):
        span = find_definition_span(original, function)
        if span is None:
            raise ValueError("Could not locate source definition for " + function)
        replacements.append((span, Path(candidate_path).read_text().strip() + "\n"))

    staged = original
    for span, candidate in sorted(replacements, reverse=True):
        staged = staged[:span[0]] + candidate + staged[span[1]:]
    staged_source = output_dir / "uniq.recovered.c"
    staged_source.write_text(staged)

    object_path = output_dir / "uniq.recovered.o"
    compiler = [
        args.clang, "-I.", "-I..", "-I./lib", "-Ilib", "-I../lib", "-Isrc", "-I../src",
        "-Wno-format-extra-args", "-Wno-implicit-const-int-float-conversion",
        "-Wno-tautological-constant-out-of-range-compare", "-O3", "-fno-omit-frame-pointer",
        "-c", str(staged_source), "-o", str(object_path),
    ]
    compile_result = subprocess.run(compiler, cwd=build_dir, text=True, capture_output=True)
    (output_dir / "compile.stdout").write_text(compile_result.stdout)
    (output_dir / "compile.stderr").write_text(compile_result.stderr)
    compile_result.check_returncode()

    executable = output_dir / "uniq.recovered"
    link = [
        args.clang, "-O3", "-fno-omit-frame-pointer", "-o", str(executable), str(object_path),
        *args.original_object,
        "src/libver.a", args.coreutils_library, args.coreutils_library,
    ]
    link_result = subprocess.run(link, cwd=build_dir, text=True, capture_output=True)
    (output_dir / "link.stdout").write_text(link_result.stdout)
    (output_dir / "link.stderr").write_text(link_result.stderr)
    link_result.check_returncode()
    print("executable: " + str(executable))


if __name__ == "__main__":
    main()
