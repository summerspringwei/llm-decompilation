"""
Remill-based decompilation: Load dataset, lift assembly to LLVM IR using Remill,
verify with exebench, and report success rates.

Similar to gemini_decompilation.py but uses Remill instead of LLM.
"""

import os
import argparse
import pickle
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Optional

from datasets import load_from_disk

from utils.mylogger import logger
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from models.gemini.remill.remill_lifter import lift_assembly_to_llvm_ir, DEFAULT_REMILL_BUILD

HOME_DIR = os.path.expanduser("~")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Remill-based assembly-to-LLVM-IR lifting and evaluate on exebench"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sampled_dataset_with_loops_and_only_one_bb_164",
        help="Dataset name to use",
    )
    parser.add_argument(
        "--remill_build",
        type=str,
        default=DEFAULT_REMILL_BUILD,
        help="Path to Remill build directory (e.g., ~/Software/remill/build)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes for parallel execution",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    return parser.parse_args()


args = parse_args()
remill_build = os.path.expanduser(args.remill_build)

# Dataset paths (same as gemini_decompilation)
dataset_pairs = {
    "sampled_dataset_with_loops_and_only_one_bb_164": (
        f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
        f"{HOME_DIR}/Projects/validation/remill/sample_only_one_bb",
    ),
    "sampled_dataset_without_loops_164": (
        f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_without_loops_164",
        f"{HOME_DIR}/Projects/validation/remill/sample_without_loops",
    ),
    "sampled_dataset_with_loops_164": (
        f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_164",
        f"{HOME_DIR}/Projects/validation/remill/sample_loops",
    ),
}


@dataclass
class RemillDecompileResult:
    """Result of Remill lifting for one record."""

    idx: int
    predict_compile_success: bool
    predict_execution_success: bool
    target_compile_success: bool
    target_execution_success: bool
    error_msg: Optional[str] = None


def process_one_record(
    record: dict,
    idx: int,
    output_dir: str,
    remill_build: str,
) -> RemillDecompileResult:
    """
    Process one dataset record: lift assembly with Remill, verify target, and evaluate.
    """
    sample_dir = os.path.join(output_dir, f"sample_{idx}")
    os.makedirs(sample_dir, exist_ok=True)

    asm_code = record["asm"]["code"][-1]
    target_llvm_ir = record["llvm_ir"]["code"][-1]

    # 1. Remill lift: assembly -> LLVM IR -> compiled assembly
    lift_ok, lifted_ir, remill_assembly, lift_err = lift_assembly_to_llvm_ir(
        asm_code,
        sample_dir,
        remill_build=remill_build,
        remove_comments=True,
    )

    predict_compile_success = lift_ok
    predict_execution_success = False

    if lift_ok and lifted_ir and remill_assembly:
        # Remill produces assembly with sub_0(State*, i64, Memory*) ABI.
        # exebench expects C-callable assembly. We try eval_assembly for consistency,
        # but it will typically fail because of ABI mismatch.
        try:
            predict_execution_success = eval_assembly(record, remill_assembly)
        except Exception as e:
            logger.debug(f"Record {idx}: eval_assembly failed (expected for Remill): {e}")

    # 2. Verify target (ground truth) LLVM IR
    target_compile_success, target_assembly_path, target_error_msg = compile_llvm_ir(
        target_llvm_ir, sample_dir, name_hint="target"
    )
    target_execution_success = False
    if target_compile_success and target_assembly_path:
        try:
            with open(target_assembly_path, "r") as f:
                target_assembly = f.read()
            target_execution_success = eval_assembly(record, target_assembly)
        except Exception as e:
            logger.warning(f"Record {idx}: target eval_assembly failed: {e}")

    return RemillDecompileResult(
        idx=idx,
        predict_compile_success=predict_compile_success,
        predict_execution_success=predict_execution_success,
        target_compile_success=target_compile_success,
        target_execution_success=target_execution_success,
        error_msg=lift_err if not lift_ok else None,
    )


def _process_one_wrapper(args_tuple):
    return process_one_record(*args_tuple)


def run_remill_decompilation(
    dataset_dir: str,
    output_dir: str,
    remill_build: str,
    num_processes: int = 1,
    limit: Optional[int] = None,
) -> list:
    """
    Load dataset, run Remill lifting on each record, and return results.
    """
    dataset = load_from_disk(dataset_dir)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    os.makedirs(output_dir, exist_ok=True)

    task_args = [
        (record, idx, output_dir, remill_build)
        for idx, record in enumerate(dataset)
    ]

    if num_processes <= 1:
        results = [process_one_record(*a) for a in task_args]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(process_one_record, task_args)

    # Save results
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results


def main():
    if args.dataset_name not in dataset_pairs:
        raise ValueError(
            f"Dataset '{args.dataset_name}' not found. "
            f"Available: {list(dataset_pairs.keys())}"
        )

    dataset_dir, output_dir = dataset_pairs[args.dataset_name]
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Remill build: {remill_build}")

    results = run_remill_decompilation(
        dataset_dir,
        output_dir,
        remill_build,
        num_processes=args.num_processes,
        limit=args.limit,
    )

    # Report success rates (same format as gemini_decompilation)
    predict_compile_count = sum(1 for r in results if r.predict_compile_success)
    predict_execution_count = sum(1 for r in results if r.predict_execution_success)
    target_compile_count = sum(1 for r in results if r.target_compile_success)
    target_execution_count = sum(1 for r in results if r.target_execution_success)
    total = len(results)

    print("\n" + "=" * 60)
    print("Remill Decompilation - Overall Success Rate")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Number of predict_compile_success: {predict_compile_count} ({100*predict_compile_count/total:.1f}%)")
    print(f"Number of predict_execution_success: {predict_execution_count} ({100*predict_execution_count/total:.1f}%)")
    print(f"Number of target_compile_success: {target_compile_count} ({100*target_compile_count/total:.1f}%)")
    print(f"Number of target_execution_success: {target_execution_count} ({100*target_execution_count/total:.1f}%)")
    print("=" * 60)
    print("\nNote: Remill produces trace-based IR with (State*, i64, Memory*) ABI.")
    print("exebench expects C-callable assembly, so predict_execution_success is")
    print("typically 0 for Remill (ABI mismatch).")


if __name__ == "__main__":
    main()
