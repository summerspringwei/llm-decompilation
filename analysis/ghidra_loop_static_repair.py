"""Generate loop-aware static repair prompts using Ghidra CFG extraction.

This script applies a three-layer static repair analysis for loop-heavy
validation failures:

1. CFG and loop summaries from Ghidra basic-block extraction.
2. Semantic instruction-pattern summaries.
3. Target-vs-predicted assembly feature diffs.

It writes one analysis report and one LLM repair prompt per selected sample.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import GhidraConfig  # noqa: E402
from models.gemini.summarize_validation import (  # noqa: E402
    ResultUnpickler,
    get_attr_or_key,
    record_prediction_success,
    record_target_success,
)
from utils.evaluate_exebench import compile_llvm_ir  # noqa: E402


BRANCH_MNEMONICS = {
    "ja",
    "jae",
    "jb",
    "jbe",
    "jc",
    "je",
    "jg",
    "jge",
    "jl",
    "jle",
    "jna",
    "jnae",
    "jnb",
    "jnbe",
    "jnc",
    "jne",
    "jno",
    "jnp",
    "jns",
    "jo",
    "jp",
    "jpe",
    "jpo",
    "js",
    "jmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_dir", type=Path, required=True)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated sample indices. Defaults to all failed predictions.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Defaults to <validation_dir>/ghidra_loop_static_repair.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ghidra_timeout", type=int, default=180)
    return parser.parse_args()


def run_command(command: list[str], timeout: int) -> tuple[int, str, str]:
    output = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return output.returncode, output.stdout, output.stderr


def load_results(validation_dir: Path) -> list[Any]:
    with (validation_dir / "results.pkl").open("rb") as f:
        return ResultUnpickler(f).load()


def failed_prediction_indices(results: list[Any]) -> list[int]:
    indices = []
    for idx, record in enumerate(results):
        pred = record_prediction_success(record)
        target = record_target_success(record)
        if target[1] and not pred[1]:
            indices.append(idx)
    return indices


def retry_items(record: Any) -> list[tuple[int, Any]]:
    retry = get_attr_or_key(record, "retry_response_validation", {}) or {}
    if isinstance(retry, dict):
        return sorted((k, v) for k, v in retry.items() if k != -1)
    return list(enumerate(retry))


def choose_prediction(record: Any) -> tuple[int, int, Any] | None:
    """Choose the latest compilable non-success prediction."""
    for retry_no, rv in reversed(retry_items(record)):
        evals = get_attr_or_key(rv, "predict_evaluation_results_list", []) or []
        for choice_idx, er in enumerate(evals):
            if get_attr_or_key(er, "compile_success", False):
                return retry_no, choice_idx, er
    return None


def get_target_result(record: Any) -> Any | None:
    retry = get_attr_or_key(record, "retry_response_validation", {}) or {}
    if isinstance(retry, dict) and -1 in retry:
        return get_attr_or_key(retry[-1], "target_evaluation_result")
    for _retry_no, rv in retry_items(record):
        target = get_attr_or_key(rv, "target_evaluation_result")
        if target:
            return target
    return None


def ensure_object_from_ir(
    llvm_ir: str,
    output_dir: Path,
    name_hint: str,
) -> Path | None:
    success, _assembly_path, error_msg = compile_llvm_ir(
        llvm_ir,
        str(output_dir),
        name_hint=name_hint,
    )
    object_path = output_dir / f"{name_hint}.o"
    if not success or not object_path.exists():
        (output_dir / f"{name_hint}.compile_error.txt").write_text(error_msg or "")
        return None
    return object_path


def extract_cfg_with_ghidra(
    object_path: Path,
    func_name: str,
    output_json: Path,
    timeout: int,
) -> dict[str, Any] | None:
    ghidra_cfg = GhidraConfig()
    script_path = REPO_ROOT / "models" / "ghidra_decompile" / "ghidra_extract_bb.py"
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            ghidra_cfg.headless_analyzer_path,
            tmp_dir,
            f"cfg_{object_path.stem}_{func_name}",
            "-import",
            str(object_path),
            "-overwrite",
            "-postscript",
            str(script_path),
            func_name,
            str(output_json),
        ]
        retcode, stdout, stderr = run_command(cmd, timeout=timeout)
        output_json.with_suffix(".stdout.txt").write_text(stdout)
        output_json.with_suffix(".stderr.txt").write_text(stderr)
        if retcode != 0 or not output_json.exists():
            return None
    return json.loads(output_json.read_text())


def extract_function_cfg(cfg_json: dict[str, Any]) -> dict[str, Any] | None:
    for _program_path, program_data in cfg_json.items():
        if not isinstance(program_data, dict):
            continue
        for key, value in program_data.items():
            if key == "arch":
                continue
            if isinstance(value, dict) and "nodes" in value and "basic_blocks" in value:
                return value
    return None


def graph_from_cfg(func_cfg: dict[str, Any]) -> tuple[set[int], list[tuple[int, int]]]:
    nodes = {int(n) for n in func_cfg.get("nodes", [])}
    edges = [(int(src), int(dst)) for src, dst in func_cfg.get("edges", [])]
    return nodes, edges


def tarjan_scc(nodes: set[int], edges: list[tuple[int, int]]) -> list[set[int]]:
    adjacency: dict[int, list[int]] = {node: [] for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)

    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    result: list[set[int]] = []

    def strongconnect(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for succ in adjacency.get(node, []):
            if succ not in indices:
                strongconnect(succ)
                lowlinks[node] = min(lowlinks[node], lowlinks[succ])
            elif succ in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[succ])

        if lowlinks[node] == indices[node]:
            component = set()
            while True:
                succ = stack.pop()
                on_stack.remove(succ)
                component.add(succ)
                if succ == node:
                    break
            result.append(component)

    for node in nodes:
        if node not in indices:
            strongconnect(node)
    return result


def summarize_loops(func_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    nodes, edges = graph_from_cfg(func_cfg)
    edge_set = set(edges)
    components = tarjan_scc(nodes, edges)
    loops = []
    for component in components:
        self_loop = any((node, node) in edge_set for node in component)
        if len(component) <= 1 and not self_loop:
            continue
        entries = sorted(
            dst for src, dst in edges if src not in component and dst in component
        )
        exits = sorted(
            (src, dst) for src, dst in edges if src in component and dst not in component
        )
        backedges = sorted(
            (src, dst) for src, dst in edges if src in component and dst in component and dst <= src
        )
        loops.append(
            {
                "nodes": sorted(component),
                "headers": entries or [min(component)],
                "exits": exits,
                "backedges": backedges,
            }
        )
    return sorted(loops, key=lambda loop: (loop["headers"][0], len(loop["nodes"])))


def all_instructions(func_cfg: dict[str, Any]) -> list[str]:
    blocks = func_cfg.get("basic_blocks", {}) or {}
    instructions = []
    for block_addr in sorted(blocks, key=lambda item: int(item)):
        instructions.extend(blocks[block_addr].get("bb_disasm", []) or [])
    return instructions


def mnemonic(instruction: str) -> str:
    return instruction.split(None, 1)[0].lower() if instruction.strip() else ""


def extract_constants(instruction: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9_])-?(?:0x[0-9a-fA-F]+|\d+)", instruction)


def classify_memory_width(instruction: str) -> str | None:
    text = instruction.lower()
    if "xmm" in text or "ymm" in text:
        return "vector"
    if "qword" in text or re.search(r"%r[a-z0-9]+", text):
        return "qword"
    if "dword" in text or re.search(r"%e[a-z0-9]+", text):
        return "dword"
    if "word" in text:
        return "word"
    if "byte" in text or re.search(r"%[a-d][lh]", text):
        return "byte"
    return None


def memory_scale(instruction: str) -> str | None:
    match = re.search(r",\s*([1248])\)", instruction)
    return match.group(1) if match else None


def semantic_features(func_cfg: dict[str, Any]) -> dict[str, Any]:
    instructions = all_instructions(func_cfg)
    mnems = [mnemonic(inst) for inst in instructions]
    calls = [inst for inst in instructions if mnemonic(inst).startswith("call")]
    branches = [inst for inst in instructions if mnemonic(inst) in BRANCH_MNEMONICS]
    compares = [
        inst
        for inst in instructions
        if mnemonic(inst).startswith(("cmp", "test"))
    ]
    increments = [
        inst
        for inst in instructions
        if mnemonic(inst).startswith(("add", "sub", "inc", "dec", "lea"))
    ]
    loads = [
        inst
        for inst in instructions
        if mnemonic(inst).startswith("mov") and "(" in inst and ")" in inst
    ]
    stores = [
        inst
        for inst in instructions
        if mnemonic(inst).startswith("mov") and re.search(r"\([^)]*\)\s*$", inst)
    ]
    constants = Counter()
    for inst in instructions:
        constants.update(extract_constants(inst))
    memory_widths = Counter(
        width for inst in loads + stores if (width := classify_memory_width(inst))
    )
    memory_scales = Counter(
        scale for inst in loads + stores if (scale := memory_scale(inst))
    )
    return {
        "instruction_count": len(instructions),
        "mnemonic_counts": Counter(mnems),
        "calls": calls,
        "branches": branches,
        "compares": compares,
        "increments": increments,
        "loads": loads,
        "stores": stores,
        "constants": constants,
        "memory_widths": memory_widths,
        "memory_scales": memory_scales,
    }


def top_counter(counter: Counter, limit: int = 12) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def compact_list(items: list[str], limit: int = 8) -> str:
    if not items:
        return "none"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f"\n  ... {len(items) - limit} more"
    return "\n  " + "\n  ".join(shown) + suffix


def compare_features(target: dict[str, Any], predict: dict[str, Any]) -> list[str]:
    findings = []
    for name in ("calls", "branches", "compares", "increments", "loads", "stores"):
        target_set = set(target[name])
        predict_set = set(predict[name])
        missing = sorted(target_set - predict_set)[:8]
        extra = sorted(predict_set - target_set)[:8]
        if missing:
            findings.append(f"Missing target {name}: " + "; ".join(missing))
        if extra:
            findings.append(f"Extra predicted {name}: " + "; ".join(extra))

    for name in ("memory_widths", "memory_scales", "constants"):
        if target[name] != predict[name]:
            findings.append(
                f"{name} differ: target [{top_counter(target[name])}], "
                f"predict [{top_counter(predict[name])}]"
            )
    return findings or ["No obvious static feature diff found; inspect loop summaries."]


def format_loop_summary(label: str, func_cfg: dict[str, Any]) -> str:
    nodes, edges = graph_from_cfg(func_cfg)
    loops = summarize_loops(func_cfg)
    lines = [
        f"{label} CFG:",
        f"- basic blocks: {len(nodes)}",
        f"- edges: {len(edges)}",
        f"- loops/SCCs: {len(loops)}",
    ]
    for i, loop in enumerate(loops[:8]):
        lines.extend(
            [
                f"  loop {i}:",
                f"    headers: {loop['headers']}",
                f"    nodes: {loop['nodes']}",
                f"    backedges: {loop['backedges']}",
                f"    exits: {loop['exits']}",
            ]
        )
    if len(loops) > 8:
        lines.append(f"  ... {len(loops) - 8} more loops")
    return "\n".join(lines)


def format_semantic_summary(label: str, features: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{label} semantic instruction patterns:",
            f"- instruction count: {features['instruction_count']}",
            f"- mnemonic counts: {top_counter(features['mnemonic_counts'])}",
            f"- calls:{compact_list(features['calls'])}",
            f"- compares/tests:{compact_list(features['compares'])}",
            f"- branches:{compact_list(features['branches'])}",
            f"- induction/update-like instructions:{compact_list(features['increments'])}",
            f"- memory widths: {top_counter(features['memory_widths'])}",
            f"- memory scales: {top_counter(features['memory_scales'])}",
            f"- constants: {top_counter(features['constants'])}",
        ]
    )


def build_repair_prompt(
    record: Any,
    prediction: Any,
    analysis_text: str,
) -> str:
    initial_prompt = get_attr_or_key(record, "initial_prompt", "") or ""
    predicted_ir = get_attr_or_key(prediction, "llvm_ir", "") or ""
    predicted_assembly = get_attr_or_key(prediction, "assembly", "") or ""
    return f"""{initial_prompt}

The previous LLVM IR compiles but does not match the target behavior for a function with loops.

Previous LLVM IR:
```llvm
{predicted_ir}
```

Predicted assembly:
```assembly
{predicted_assembly}
```

Static Ghidra-based loop analysis:
```text
{analysis_text}
```

Use the analysis to repair the LLVM IR. Focus on loop bounds, induction variable updates,
branch conditions, memory access width/stride, calls, and loop-carried stores.
Return only the corrected LLVM IR between ```llvm and ```.
"""


def analyze_sample(
    validation_dir: Path,
    output_dir: Path,
    idx: int,
    record: Any,
    ghidra_timeout: int,
) -> None:
    row = get_attr_or_key(record, "record")
    func_info = get_attr_or_key(row, "func_info", {}) or {}
    functions = get_attr_or_key(func_info, "functions", []) or []
    first_function = functions[0] if functions else {}
    func_name = get_attr_or_key(row, "fname", "") or get_attr_or_key(
        first_function,
        "name",
        "",
    )
    selected = choose_prediction(record)
    target = get_target_result(record)
    sample_out = output_dir / f"sample_{idx}"
    sample_out.mkdir(parents=True, exist_ok=True)

    if not selected or not target:
        (sample_out / "analysis.txt").write_text("No compilable prediction or target result found.\n")
        return

    retry_no, choice_idx, prediction = selected
    target_obj = ensure_object_from_ir(
        get_attr_or_key(target, "llvm_ir", "") or "",
        sample_out / "target",
        "target",
    )
    predict_obj = ensure_object_from_ir(
        get_attr_or_key(prediction, "llvm_ir", "") or "",
        sample_out / f"retry_{retry_no}_choice_{choice_idx}",
        "predict",
    )
    if not target_obj or not predict_obj:
        (sample_out / "analysis.txt").write_text("Could not compile target or prediction to object.\n")
        return

    target_cfg_json = extract_cfg_with_ghidra(
        target_obj,
        func_name,
        sample_out / "target_cfg.json",
        ghidra_timeout,
    )
    predict_cfg_json = extract_cfg_with_ghidra(
        predict_obj,
        func_name,
        sample_out / "predict_cfg.json",
        ghidra_timeout,
    )
    if not target_cfg_json or not predict_cfg_json:
        (sample_out / "analysis.txt").write_text("Could not extract one or both CFGs with Ghidra.\n")
        return

    target_cfg = extract_function_cfg(target_cfg_json)
    predict_cfg = extract_function_cfg(predict_cfg_json)
    if not target_cfg or not predict_cfg:
        (sample_out / "analysis.txt").write_text("Could not locate function CFG in Ghidra JSON.\n")
        return

    target_features = semantic_features(target_cfg)
    predict_features = semantic_features(predict_cfg)
    findings = compare_features(target_features, predict_features)
    analysis_text = "\n\n".join(
        [
            f"sample index: {idx}",
            f"function: {func_name}",
            f"path: {get_attr_or_key(row, 'path', '')}",
            f"prediction used: retry {retry_no}, choice {choice_idx}",
            "Layer 1: CFG and loop summary",
            format_loop_summary("Target", target_cfg),
            format_loop_summary("Prediction", predict_cfg),
            "Layer 2: semantic instruction-pattern summary",
            format_semantic_summary("Target", target_features),
            format_semantic_summary("Prediction", predict_features),
            "Layer 3: target-vs-prediction feature differences",
            "\n".join(f"- {finding}" for finding in findings),
            "Repair guidance",
            "- If loop counts differ, fix the loop condition or induction update.",
            "- If memory widths/scales differ, fix load/store type or getelementptr stride.",
            "- If compare/branch patterns differ, fix signedness and exit condition.",
            "- If calls differ, preserve external helper calls and their argument order.",
        ]
    )
    (sample_out / "analysis.txt").write_text(analysis_text)
    (sample_out / "repair_prompt.txt").write_text(
        build_repair_prompt(record, prediction, analysis_text)
    )
    print(f"wrote {sample_out / 'analysis.txt'}")


def main() -> None:
    args = parse_args()
    results = load_results(args.validation_dir)
    if args.indices.strip():
        indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
    else:
        indices = failed_prediction_indices(results)
    if args.limit > 0:
        indices = indices[: args.limit]
    output_dir = args.output_dir or (args.validation_dir / "ghidra_loop_static_repair")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        analyze_sample(
            args.validation_dir,
            output_dir,
            idx,
            results[idx],
            args.ghidra_timeout,
        )


if __name__ == "__main__":
    main()
