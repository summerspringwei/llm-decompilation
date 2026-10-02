"""Codex-internal C decompilation artifact/evaluation runner.

This entry point evaluates C-function predictions without calling any model
API.  It mirrors the validation layout used by the LLVM IR runner, but reads
``sample_<idx>.c`` predictions, compiles them to LLVM IR/assembly, extracts the
predicted function assembly, and verifies it with ``eval_assembly_with_details``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from config import HOME_DIR
from models.assembly_analyzer.extract_asm_function_define import split_elf_functions
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly_with_details


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_without_loops_164",
    ),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
}


@dataclass
class CEvaluationResult:
    compile_success: bool
    execution_success: bool
    error_msg: str = ""
    c_code: str = ""
    llvm_ir: str = ""
    assembly: str = ""
    execution_details: dict | None = None

    def __post_init__(self) -> None:
        details = self.execution_details or {}
        self.execution_details = details
        self.pass_count = int(details.get("pass_count", 0) or 0)
        self.total_count = int(details.get("total_count", 0) or 0)


@dataclass
class CRecordResult:
    idx: int
    predict_evaluation: CEvaluationResult
    target_evaluation: CEvaluationResult

    def predict_has_compile_and_execution_success(self) -> tuple[bool, bool]:
        return (
            self.predict_evaluation.compile_success,
            self.predict_evaluation.execution_success,
        )

    def target_has_compile_and_execution_success(self) -> tuple[bool, bool]:
        return (
            self.target_evaluation.compile_success,
            self.target_evaluation.execution_success,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local C function predictions without model API calls."
    )
    parser.add_argument("--dataset_name", default="sampled_dataset_with_loops_164")
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--opt", default="-O2")
    return parser.parse_args()


def default_output_dir(dataset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        HOME_DIR,
        "Projects",
        "validation",
        "codex",
        f"{timestamp}_{dataset_name}_codex-c-no-api",
    )


def selected_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices.strip():
        return [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if args.max_samples > 0:
        return list(range(min(args.max_samples, dataset_len)))
    return list(range(dataset_len))


def prediction_path(predictions_dir: Path, idx: int) -> Path:
    candidates = [predictions_dir / f"sample_{idx}.c", predictions_dir / f"{idx}.c"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No C prediction found for sample {idx} in {predictions_dir}")


def _source_for_compile(record: dict, c_code: str) -> str:
    deps = record["synth_deps"].replace("typedef int bool;", "")
    return deps + "\n" + c_code.strip() + "\n"


def _assert_function_present(full_assembly: str, fname: str) -> None:
    functions = split_elf_functions(full_assembly)
    if fname not in functions:
        raise KeyError(f"Compiled assembly does not contain function {fname}")


def compile_c_prediction(
    record: dict,
    c_code: str,
    sample_dir: Path,
    clang: str,
    opt: str,
) -> tuple[bool, str, str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    c_path = sample_dir / "predict.c"
    ll_path = sample_dir / "predict.ll"
    full_s_path = sample_dir / "predict.full.s"
    s_path = sample_dir / "predict.s"
    error_path = sample_dir / "predict.error"

    compile_source = _source_for_compile(record, c_code)
    c_path.write_text(c_code)
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = Path(tmp_dir) / "predict_with_deps.c"
        src_path.write_text(compile_source)
        base_cmd = [
            clang,
            "-std=c11",
            opt,
            "-fcommon",
            "-fno-asynchronous-unwind-tables",
            "-Wno-int-conversion",
            "-Wno-implicit-function-declaration",
            "-Wno-incompatible-pointer-types",
        ]
        ll_ret = subprocess.run(
            base_cmd + ["-S", "-emit-llvm", str(src_path), "-o", str(ll_path)],
            capture_output=True,
            text=True,
        )
        asm_ret = subprocess.run(
            base_cmd + ["-S", str(src_path), "-o", str(full_s_path)],
            capture_output=True,
            text=True,
        )

    if ll_ret.returncode != 0 or asm_ret.returncode != 0:
        error_msg = (ll_ret.stderr + "\n" + asm_ret.stderr).strip()
        error_path.write_text(error_msg)
        return False, "", ll_path.read_text() if ll_path.exists() else "", error_msg

    full_assembly = full_s_path.read_text()
    try:
        _assert_function_present(full_assembly, record["fname"])
    except Exception as exc:
        error_msg = str(exc)
        error_path.write_text(error_msg)
        return False, "", ll_path.read_text(), error_msg

    s_path.write_text(full_assembly)
    return True, full_assembly, ll_path.read_text(), ""


def evaluate_c_prediction(
    record: dict,
    idx: int,
    c_code: str,
    output_dir: Path,
    clang: str,
    opt: str,
) -> CRecordResult:
    sample_dir = output_dir / f"sample_{idx}"
    predict_compile, predict_asm, predict_ir, predict_error = compile_c_prediction(
        record, c_code, sample_dir, clang, opt
    )
    predict_details = {}
    predict_exec = False
    if predict_compile:
        predict_details = eval_assembly_with_details(record, predict_asm)
        predict_exec = bool(predict_details["success"])

    target_compile, target_asm_path, target_error = compile_llvm_ir(
        record["llvm_ir"]["code"][-1], str(sample_dir), name_hint="target"
    )
    target_asm = ""
    target_details = {}
    target_exec = False
    if target_compile:
        target_asm = Path(target_asm_path).read_text()
        target_details = eval_assembly_with_details(record, target_asm)
        target_exec = bool(target_details["success"])

    if predict_exec:
        (sample_dir / "correct.c").write_text(c_code)

    return CRecordResult(
        idx=idx,
        predict_evaluation=CEvaluationResult(
            predict_compile,
            predict_exec,
            predict_error,
            c_code,
            predict_ir,
            predict_asm,
            predict_details,
        ),
        target_evaluation=CEvaluationResult(
            target_compile,
            target_exec,
            target_error,
            record["func_def"],
            record["llvm_ir"]["code"][-1],
            target_asm,
            target_details,
        ),
    )


def summarize(results: list[CRecordResult]) -> str:
    pred_compile = pred_exec = target_compile = target_exec = 0
    for result in results:
        pc, pe = result.predict_has_compile_and_execution_success()
        tc, te = result.target_has_compile_and_execution_success()
        pred_compile += int(pc)
        pred_exec += int(pe)
        target_compile += int(tc)
        target_exec += int(te)
    return (
        f"predict_compile_success: {pred_compile}\n"
        f"predict_execution_success: {pred_exec}\n"
        f"target_compile_success: {target_compile}\n"
        f"target_execution_success: {target_exec}\n"
    )


def main() -> None:
    args = parse_args()
    dataset_path = DATASET_PATHS[args.dataset_name]
    dataset = load_from_disk(dataset_path)
    indices = selected_indices(len(dataset), args)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir or default_output_dir(args.dataset_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

    results: list[CRecordResult] = []
    for idx in tqdm(indices, desc="Evaluating Codex C predictions", unit="sample"):
        record = dataset[idx]
        pred_path = prediction_path(predictions_dir, idx)
        c_code = pred_path.read_text()
        (output_dir / "predictions" / f"sample_{idx}.c").write_text(c_code)
        (output_dir / f"response_{idx}.txt").write_text(c_code)
        result = evaluate_c_prediction(record, idx, c_code, output_dir, args.clang, args.opt)
        results.append(result)

    with (output_dir / "results.pkl").open("wb") as f:
        pickle.dump(results, f)
    summary = summarize(results)
    (output_dir / "summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
