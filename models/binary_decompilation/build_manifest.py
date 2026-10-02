"""Create an auditable source/object manifest for a complete ELF decompilation.

The manifest separates compiler/linker startup code from functions that can be
compared against project source. It does not use the source to generate a
decompilation; source paths are validation metadata only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from models.binary_decompilation.export_source_references import find_definition


RUNTIME_NAMES = {
    "_init", "_fini", "_start", "deregister_tm_clones", "register_tm_clones",
    "__do_global_dtors_aux", "frame_dummy", "atexit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map binary functions to source/object metadata.")
    parser.add_argument("--profile", required=True, help="program_profile.json from analyze_binary")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--program-object",
        help="Primary program object relative to --build-dir (default: src/<binary-name>.o).",
    )
    return parser.parse_args()


def nm_definitions(paths: list[Path]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        command = ["nm", "-A", "--defined-only", str(path)]
        output = subprocess.run(command, text=True, check=True, capture_output=True).stdout
        for line in output.splitlines():
            fields = line.rsplit(None, 1)
            if len(fields) != 2:
                continue
            location, name = fields
            if re.search(r"\s[Uu]\s", location):
                continue
            result.setdefault(name, []).append(location)
    return result


def source_candidates(location: str, source_dir: Path) -> list[str]:
    # nm -A emits either ``object:address type`` or
    # ``archive:member:address type``; in both forms the object is penultimate.
    object_name = location.split(":")[-2]
    stem = Path(object_name).stem
    for prefix in ("libcoreutils_a-", "libsinglebin_", "libcksum_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    candidates = sorted(source_dir.rglob(stem + ".c"))
    return [str(candidate) for candidate in candidates]


def source_definitions(candidates: list[str], name: str) -> list[str]:
    """Keep only implementation files that contain NAME's definition."""
    return [path for path in candidates if find_definition(Path(path).read_text(), name) is not None]


def main() -> None:
    args = parse_args()
    profile = json.loads(Path(args.profile).read_text())
    build_dir = Path(args.build_dir)
    source_dir = Path(args.source_dir)
    binary_name = Path(profile["binary"]).name
    program_object = (build_dir / args.program_object if args.program_object
                      else build_dir / "src" / (binary_name + ".o"))
    definitions = nm_definitions([
        program_object,
        build_dir / "src" / "version.o",
        build_dir / "lib" / "libcoreutils.a",
    ])

    manifest = []
    for function in profile["functions"]:
        name = function["name"]
        locations = definitions.get(name, [])
        candidates = [candidate for location in locations for candidate in source_candidates(location, source_dir)]
        candidates = source_definitions(sorted(set(candidates)), name)
        runtime = name in RUNTIME_NAMES
        manifest.append({
            "entry": function["entry"],
            "name": name,
            "memory_block": function["memory_block"],
            "calls": function["calls"],
            "object_definitions": locations,
            "source_candidates": sorted(set(candidates)),
            "runtime_scaffolding": runtime,
            "validation_status": ("not_applicable_runtime" if runtime else
                                  "pending" if candidates else
                                  "not_applicable_no_source_definition"),
        })

    output = {
        "binary": profile["binary"],
        "total_functions": len(manifest),
        "runtime_scaffolding": sum(item["runtime_scaffolding"] for item in manifest),
        "source_backed_functions": sum(bool(item["source_candidates"]) for item in manifest),
        "functions": manifest,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print("total_functions: %d" % output["total_functions"])
    print("source_backed_functions: %d" % output["source_backed_functions"])
    print("runtime_scaffolding: %d" % output["runtime_scaffolding"])


if __name__ == "__main__":
    main()
