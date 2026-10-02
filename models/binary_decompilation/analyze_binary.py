"""Extract a reusable Ghidra call graph and retrieval hints for an ELF binary.

The generated order is bottom-up: all direct internal callees precede their
callers. Recursive components are represented as one layer and must be handled
as a group by a downstream decompiler.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import requests
from qdrant_client import QdrantClient

from config import GhidraConfig, HOME_DIR
from utils.preprocessing_assembly import preprocessing_assembly


DEFAULT_TRAIN_ROOT = os.path.join(
    HOME_DIR,
    "Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an ELF binary with Ghidra.")
    parser.add_argument("--binary", required=True, help="Path to an ELF executable or shared library.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ghidra-home", default=GhidraConfig().home_dir)
    parser.add_argument("--decompile", action="append", default=[], help="Function name to export as Ghidra C; repeatable.")
    parser.add_argument("--retrieve", action="store_true", help="Query Qdrant for each selected function.")
    parser.add_argument("--embedding-backend", choices=("text", "hermessim"), default="text")
    parser.add_argument("--embedding-url", default="http://localhost:8001/embed/batch")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-template", default="train_synth_rich_io_filtered_{idx}_preprocessed")
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    return parser.parse_args()


def run_ghidra(binary: Path, output: Path, ghidra_home: str, functions: list[str], env=None) -> None:
    script_dir = Path(__file__).resolve().parent
    script_name = "ghidra_export_program.py"
    with tempfile.TemporaryDirectory(prefix="ghidra_binary_") as temp_dir:
        command = [
            str(Path(ghidra_home) / "support" / "analyzeHeadless"),
            temp_dir,
            "project",
            "-import",
            str(binary),
            "-scriptPath",
            str(script_dir),
            "-postScript",
            script_name,
            str(output),
            ",".join(functions),
            "-deleteProject",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=900, env=env)
    (output.parent / "ghidra.stdout").write_text(completed.stdout)
    (output.parent / "ghidra.stderr").write_text(completed.stderr)
    completed.check_returncode()


def leaf_first_layers(functions: list[dict]) -> list[list[str]]:
    internal = {function["entry"] for function in functions if not function["external"]}
    callees = {
        function["entry"]: {
            call["entry"] for call in function["calls"] if not call["external"] and call["entry"] in internal
        }
        for function in functions
        if function["entry"] in internal
    }
    remaining = set(internal)
    layers: list[list[str]] = []
    while remaining:
        leaves = sorted(entry for entry in remaining if not (callees[entry] & remaining))
        if not leaves:
            # A remaining strongly connected component is emitted together.
            layers.append(sorted(remaining))
            break
        layers.append(leaves)
        remaining.difference_update(leaves)
    return layers


def source_signature(c_code: str) -> str:
    match = re.search(r"^[^{]+\{", c_code, flags=re.MULTILINE | re.DOTALL)
    return match.group(0).rstrip("{").strip() if match else ""


def objdump_function_assembly(binary: Path, function_name: str) -> str:
    """Return a standalone GNU-as fragment for one objdump-disassembled function."""
    result = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "--disassemble=" + function_name, str(binary)],
        text=True,
        capture_output=True,
        check=True,
    )
    instruction_re = re.compile(r"^\s*([0-9a-f]+):\s*(.+)$")
    target_re = re.compile(r"\b([0-9a-f]+)\s+<([^>]+)>")
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        match = instruction_re.match(line)
        if match:
            rows.append((match.group(1), match.group(2)))
    if not rows:
        raise ValueError("objdump did not find function " + function_name)

    local_addresses = {address for address, _ in rows}

    def target_replacement(match: re.Match[str]) -> str:
        address, symbol = match.groups()
        if address in local_addresses:
            return ".L" + function_name + "_" + address
        symbol = symbol.split("+")[0]
        return symbol.replace("@plt", "@PLT")

    output = [".text", ".globl " + function_name, ".type " + function_name + ", @function", function_name + ":"]
    for address, instruction in rows:
        instruction = instruction.split("#", 1)[0].rstrip()
        instruction = target_re.sub(target_replacement, instruction)
        output.append(".L" + function_name + "_" + address + ":")
        output.append("\t" + instruction)
    output.append(".size " + function_name + ", .-" + function_name)
    return "\n".join(output) + "\n"


def embed_selected_functions(selected: list[dict], binary: Path, args: argparse.Namespace) -> list[list[float]]:
    if args.embedding_backend == "text":
        payload = [preprocessing_assembly(function["assembly"], remove_comments=True) for function in selected]
        response = requests.post(args.embedding_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["embeddings"]

    payload = [objdump_function_assembly(binary, function["name"]) for function in selected]
    response = requests.post(args.embedding_url, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    success_indices = result["success_indices"]
    if success_indices != list(range(len(selected))):
        raise RuntimeError("HermesSim failed to embed selected functions: " + repr(success_indices))
    return result["embeddings"]


def retrieve_examples(functions: list[dict], binary: Path, args: argparse.Namespace) -> dict[str, list[dict]]:
    selected = [function for function in functions if function["name"] in set(args.decompile)]
    if not selected:
        return {}
    vectors = embed_selected_functions(selected, binary, args)
    qdrant = QdrantClient(args.qdrant_host, port=args.qdrant_port)

    from datasets import load_from_disk
    shards = [load_from_disk(os.path.join(args.train_root, "train_synth_rich_io_filtered_%d_llvm_extract_func_ir_assembly_O2_llvm_diff" % shard)) for shard in range(8)]
    examples: dict[str, list[dict]] = {}
    for function, vector in zip(selected, vectors):
        hits = []
        for shard in range(8):
            hit = qdrant.search(args.collection_template.format(idx=shard), vector, limit=1)[0]
            source = shards[shard][int(hit.payload["id"])]
            hits.append({
                "score": float(hit.score),
                "collection": args.collection_template.format(idx=shard),
                "dataset_index": int(hit.payload["id"]),
                "function_name": source["fname"],
                "signature": source_signature(source["func_def"]),
            })
        examples[function["entry"]] = sorted(hits, key=lambda hit: hit["score"], reverse=True)[:3]
    return examples


def main() -> None:
    args = parse_args()
    binary = Path(args.binary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "ghidra_export.json"
    run_ghidra(binary, export_path, args.ghidra_home, args.decompile)
    exported = json.loads(export_path.read_text())
    functions = exported["functions"]
    by_entry = {function["entry"]: function for function in functions}
    retrieval = retrieve_examples(functions, binary, args) if args.retrieve else {}
    layers = leaf_first_layers(functions)
    profile = {
        "binary": str(binary),
        "functions": functions,
        "leaf_first_layers": [[{"entry": entry, "name": by_entry[entry]["name"]} for entry in layer] for layer in layers],
        "decompiled": exported["decompiled"],
        "retrieval_examples": retrieval,
    }
    (output_dir / "program_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    for name, c_code in exported["decompiled"].items():
        (output_dir / (name + ".ghidra.c")).write_text(c_code + "\n")
    print("functions: %d" % len(functions))
    print("leaf-first layers: %d" % len(layers))
    print("output_dir: %s" % output_dir)


if __name__ == "__main__":
    main()
