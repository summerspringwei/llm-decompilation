"""Replay prior angr retry failures against patched trace/prompt code.

This is a focused regression harness for validation directories that already
contain ``results.pkl`` and dumped retry prompts.  It does not rerun dataset
loading or the full decompilation pipeline.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import types
from pathlib import Path
from typing import Any


def install_fake_angr_debugger() -> None:
    """Allow importing exebench in environments without the angr package."""
    module = types.ModuleType("exebench.angr_debugger")

    def dump_x86_instructions_with_registers(**kwargs):
        start = kwargs.get("start_function_name", "<unknown>")
        exe_path = kwargs.get("exe_path", "<unknown>")
        return [
            f"0x0: trace_marker_for {start} ; exe_path={exe_path}",
            "0x1: ret",
        ]

    module.dump_x86_instructions_with_registers = dump_x86_instructions_with_registers
    sys.modules["exebench.angr_debugger"] = module

REPO_ROOT = Path(__file__).resolve().parents[1]
EXEBENCH_ROOT = Path("/data1/xiachunwei/Projects/exebench")
for path in (REPO_ROOT, EXEBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "--real_angr" not in sys.argv:
    install_fake_angr_debugger()

import exebench  # noqa: E402
from models.gemini.summarize_validation import ResultUnpickler  # noqa: E402
from utils.prompt_builder import (  # noqa: E402
    build_compile_error_prompt,
    build_execution_error_prompt_with_angr_trace,
)


PREDICT_ASM_RE = re.compile(
    r"I compiled the generated LLVM IR into the following assembly code:\n"
    r"```assembly\n(.*?)\n```\nand executed it\.",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["20:0", "65:0", "80:0", "99:9"],
        help="Cases in idx:retry format.",
    )
    parser.add_argument(
        "--use_gpp",
        action="store_true",
        help="Use exebench._DefaultAssembler for environments without clang++.",
    )
    parser.add_argument(
        "--real_angr",
        action="store_true",
        help="Use the real exebench.angr_debugger module instead of a fake trace stub.",
    )
    parser.add_argument("--write_prompts", type=Path)
    return parser.parse_args()


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_builtin(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_builtin(v) for k, v in obj.__dict__.items()}
    return obj


def load_results(validation_dir: Path) -> list[Any]:
    with (validation_dir / "results.pkl").open("rb") as f:
        return ResultUnpickler(f).load()


def load_prompt(validation_dir: Path, idx: int, retry: int) -> str:
    path = validation_dir / f"prompt_{idx}_retry_{retry}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def extract_predict_assembly(prompt: str) -> str:
    match = PREDICT_ASM_RE.search(prompt)
    if not match:
        raise ValueError("Could not find generated assembly block in retry prompt")
    return match.group(1)


def pick_predict_llvm_ir(record: Any, retry: int, fallback_prompt: str) -> str:
    prev = record.retry_response_validation[retry - 1]
    for result in prev.predict_evaluation_results_list:
        if result.compile_success and not result.execution_success and result.llvm_ir:
            return result.llvm_ir
    match = re.search(r"```llvm\n(.*?)\n```", fallback_prompt, re.DOTALL)
    return match.group(1) if match else ""


def replay_case(validation_dir: Path, results: list[Any], idx: int, retry: int) -> dict[str, Any]:
    record = results[idx]
    row = to_builtin(record.record)
    retry_prompt = load_prompt(validation_dir, idx, retry)
    predict_assembly = extract_predict_assembly(retry_prompt)
    prev = record.retry_response_validation[retry - 1]
    target_assembly = (
        getattr(prev.target_evaluation_result, "assembly", None)
        or row["asm"]["code"][-1]
    )
    target_trace, predict_trace, link_error = exebench.get_angr_traces(
        row,
        target_assembly,
        predict_assembly,
        max_insts=4,
        include_error=True,
    )

    predict_llvm_ir = pick_predict_llvm_ir(record, retry, retry_prompt)
    if target_trace.strip() and predict_trace.strip():
        next_prompt = build_execution_error_prompt_with_angr_trace(
            record.initial_prompt,
            predict_llvm_ir,
            predict_assembly,
            target_trace,
            predict_trace,
        )
        prompt_kind = "angr_trace"
    elif link_error.strip():
        next_prompt = build_compile_error_prompt(
            record.initial_prompt,
            predict_llvm_ir,
            (
                "The LLVM IR compiled to assembly, but the generated assembly "
                "could not be linked into the ExeBench wrapper for angr tracing.\n\n"
                f"{link_error}"
            ),
        )
        prompt_kind = "link_error"
    else:
        next_prompt = ""
        prompt_kind = "empty"

    return {
        "idx": idx,
        "retry": retry,
        "target_trace_lines": len(target_trace.splitlines()),
        "predict_trace_lines": len(predict_trace.splitlines()),
        "link_error_len": len(link_error),
        "prompt_kind": prompt_kind,
        "prompt_len": len(next_prompt),
        "link_error_head": link_error[:300].replace("\n", "\\n"),
        "next_prompt": next_prompt,
    }


def main() -> None:
    args = parse_args()
    if args.use_gpp:
        exebench.LLVMAssembler = exebench._DefaultAssembler

    results = load_results(args.validation_dir)
    prompt_dir = args.write_prompts
    if prompt_dir:
        prompt_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        idx_raw, retry_raw = case.split(":", 1)
        report = replay_case(args.validation_dir, results, int(idx_raw), int(retry_raw))
        if prompt_dir and report["next_prompt"]:
            out = prompt_dir / f"next_prompt_{report['idx']}_retry_{report['retry']}.txt"
            out.write_text(report["next_prompt"])
        print(
            "case={idx}:{retry} kind={prompt_kind} "
            "target_trace_lines={target_trace_lines} "
            "predict_trace_lines={predict_trace_lines} "
            "link_error_len={link_error_len} prompt_len={prompt_len}".format(**report)
        )
        if report["link_error_head"]:
            print(f"  link_error_head={report['link_error_head']}")


if __name__ == "__main__":
    main()
