"""Batch decompilation entry-point using the Anthropic Claude API.

Mirrors ``models/gemini/gemini_decompilation.py`` but uses a Claude client
instead of an OpenAI-compatible endpoint.  All result files are written under
``~/Projects/validation/claude/`` (or an explicit ``--output_dir``).

Usage::

    python -m models.claude.claude_decompilation \\
        --dataset_name sampled_dataset_with_loops_164 \\
        --model claude-sonnet-4-6 \\
        --num_generate 4 \\
        --num_processes 1
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import sys
from datetime import datetime
from multiprocessing import Pool

import faulthandler
import signal

from datasets import load_from_disk
from tqdm import tqdm

from config import DecompilationConfig, HOME_DIR
from models.claude.claude_client import ClaudeClient, ClaudeCliClient
from models.gemini.llm_decompiler import LLMDecompileRecord, EvaluationResult
from utils.logging_config import get_logger, setup_logging
from utils.prompt_builder import build_execution_error_prompt, build_compile_error_prompt
from utils.prompt_type import PromptType

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subclass to support "basic" prompt type in fix loop
# ---------------------------------------------------------------------------


class ClaudeDecompileRecord(LLMDecompileRecord):
    """Extends LLMDecompileRecord to support PromptType.BASIC in fix prompts."""

    def prepare_compile_fix_prompt(self, retry_count: int) -> str:
        from utils.llm_response_parser import extract_llvm_code_from_response

        prev = self.retry_response_validation[retry_count - 1]
        has_compile_success = any(
            r.compile_success for r in prev.predict_evaluation_results_list
        )
        predict_list = extract_llvm_code_from_response(prev.response)

        if not has_compile_success:
            predict = ""
            error_msg = ""
            for choice_idx, r in enumerate(prev.predict_evaluation_results_list):
                if r.error_msg and r.error_msg.strip():
                    error_msg = r.error_msg.strip()
                    predict = predict_list[choice_idx]
                    if isinstance(predict, list) and len(predict) > 0:
                        predict = predict[0]
                    break
            if not predict:
                predict = predict_list[0]
            from utils.prompt_builder import build_llvm_syntax_repair_prompt
            return build_llvm_syntax_repair_prompt(self.initial_prompt, predict, error_msg)

        if prev.get_num_compile_success() == 1:
            best = prev.get_first_compile_success_evaluation_result()
        else:
            best = self.get_best_retry_candidate(retry_count)

        from utils.preprocessing_assembly import preprocessing_assembly
        predict_llvm_ir = best.llvm_ir
        predict_assembly = preprocessing_assembly(
            best.assembly, remove_comments=self.config.remove_comments
        )

        # Basic prompt type: treat the same as SIMILAR_RECORD for fix prompts.
        if self.prompt_type in (
            PromptType.BASIC,
            PromptType.GHIDRA_DECOMPILE,
            PromptType.SIMILAR_RECORD,
        ):
            return build_execution_error_prompt(
                self.initial_prompt,
                predict_llvm_ir,
                predict_assembly,
                best.execution_details,
            )
        # Delegate all other types to the parent implementation.
        return super().prepare_compile_fix_prompt(retry_count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Claude-based decompilation on an ExeBench dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sampled_dataset_with_loops_164",
        help="Dataset name (must match a key in DATASET_PAIRS).",
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument(
        "--use_cli",
        action="store_true",
        default=True,
        help="Use the `claude -p` CLI (session auth, no API key needed). Default: True.",
    )
    parser.add_argument(
        "--no_use_cli",
        dest="use_cli",
        action="store_false",
        help="Use the Anthropic SDK instead of the CLI (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--prompt-type",
        dest="prompt_type",
        type=str,
        default="basic",
        help="Prompt strategy: basic | in-context-learning | ghidra-decompile",
    )
    parser.add_argument("--num_generate", type=int, default=4)
    parser.add_argument("--num_retry", type=int, default=5)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument(
        "--max_tokens", type=int, default=8192,
        help="Max tokens per Claude response.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated original dataset indices to run, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Run only the first N samples from the selected dataset (0 = all).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Override the validation output directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_client = None
_model_name = None
_config: DecompilationConfig | None = None


@contextlib.contextmanager
def _redirect_worker_output(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(f"\n===== subprocess pid={os.getpid()} start =====\n")
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            log_file.write(f"===== subprocess pid={os.getpid()} end =====\n")


def _decompile_func(record, idx: int) -> LLMDecompileRecord:
    faulthandler.register(signal.SIGUSR1)

    sample_dir = os.path.join(_config.output_dir, f"sample_{idx}")
    log_path = os.path.join(sample_dir, "subprocess.log")
    with _redirect_worker_output(log_path):
        logger.info("Starting decompilation worker for sample %d", idx)
        llm_record = ClaudeDecompileRecord(
            record=record,
            idx=idx,
            config=_config,
            llm_client=_client,
            model_name=_model_name,
            rag_search=None,  # RAG disabled for Claude baseline run
        )
        llm_record.get_initial_prompt()
        llm_record.decompile_and_evaluate(llm_record.initial_prompt, -1)
        llm_record.correct_one()
        llm_record.finalize()
        logger.info("Finished decompilation worker for sample %d", idx)
        return llm_record


def _decompile_func_from_args(args) -> tuple[int, LLMDecompileRecord]:
    record, idx = args
    return idx, _decompile_func(record, idx)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_decompilation(
    dataset,
    config: DecompilationConfig,
    sample_indices: list[int] | None = None,
    max_samples: int = 0,
) -> list[LLMDecompileRecord]:
    output_dir = config.output_dir
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")

    if sample_indices:
        args_list = [(dataset[idx], idx) for idx in sample_indices]
    elif max_samples and max_samples > 0:
        args_list = [(dataset[idx], idx) for idx in range(min(max_samples, len(dataset)))]
    else:
        args_list = [(record, idx) for idx, record in enumerate(dataset)]

    indexed_results: list[tuple[int, LLMDecompileRecord]] = []
    with Pool(processes=config.num_processes) as pool:
        for item in tqdm(
            pool.imap_unordered(_decompile_func_from_args, args_list),
            total=len(args_list),
            desc="Decompiling samples",
            unit="sample",
        ):
            indexed_results.append(item)

    results = [
        result
        for _, result in sorted(indexed_results, key=lambda item: item[0])
    ]

    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    pred_compile, pred_exec = 0, 0
    tgt_compile, tgt_exec = 0, 0
    for r in results:
        pc, pe = r.predict_has_compile_and_execution_success()
        if pc:
            pred_compile += 1
        if pe:
            pred_exec += 1
        tc, te = r.target_has_compile_and_execution_success()
        if tc:
            tgt_compile += 1
        if te:
            tgt_exec += 1

    logger.info("predict_compile_success: %d", pred_compile)
    logger.info("predict_execution_success: %d", pred_exec)
    logger.info("target_compile_success: %d", tgt_compile)
    logger.info("target_execution_success: %d", tgt_exec)

    return results


# ---------------------------------------------------------------------------
# Dataset mappings
# ---------------------------------------------------------------------------


def _build_dataset_pairs(
    model: str,
    num_generate: int,
    prompt_type: str,
) -> dict[str, tuple[str, str]]:
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _output_dir(subset_label: str) -> str:
        return os.path.join(
            HOME_DIR,
            "Projects",
            "validation",
            "claude",
            f"{run_timestamp}_{subset_label}_{model}-n{num_generate}-assembly-without-comments-{prompt_type}",
        )

    return {
        "sampled_dataset_with_loops_164": (
            os.path.join(
                HOME_DIR,
                "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
            ),
            _output_dir("sample_loops"),
        ),
        "sampled_dataset_with_loops_and_only_one_bb_164": (
            os.path.join(
                HOME_DIR,
                "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
            ),
            _output_dir("sample_only_one_bb"),
        ),
        "sampled_dataset_without_loops_164": (
            os.path.join(
                HOME_DIR,
                "Datasets/filtered_exebench/sampled_dataset_without_loops_164",
            ),
            _output_dir("sample_without_loops"),
        ),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    global _client, _model_name, _config

    setup_logging()
    args = parse_args()

    # Build a minimal DecompilationConfig (no RAG, no pcode for baseline).
    config = DecompilationConfig(
        model_name=args.model,
        num_generate=args.num_generate,
        num_retry=args.num_retry,
        num_processes=args.num_processes,
        prompt_type=args.prompt_type,
        fix_prompt_type="compile-fix",
        remove_comments=True,
        use_pcode=False,
        use_angr_trace=False,
        dataset_name=args.dataset_name,
    )

    _model_name = args.model
    if args.use_cli:
        logger.info("Using claude CLI backend (session auth)")
        _client = ClaudeCliClient(model=args.model, timeout_secs=config.llm_timeout)
    else:
        logger.info("Using Anthropic SDK backend")
        _client = ClaudeClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            default_max_tokens=args.max_tokens,
        )

    dataset_pairs = _build_dataset_pairs(
        model=args.model,
        num_generate=args.num_generate,
        prompt_type=args.prompt_type,
    )
    if config.dataset_name not in dataset_pairs:
        raise ValueError(
            f"Unknown dataset '{config.dataset_name}'. "
            f"Known: {sorted(dataset_pairs)}"
        )

    dataset_path, output_dir = dataset_pairs[config.dataset_name]
    if args.output_dir:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir
    _config = config

    logger.info("Output directory: %s", output_dir)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Model: %s  n=%d  retries=%d", args.model, args.num_generate, args.num_retry)

    dataset = load_from_disk(dataset_path)
    logger.info("Loaded dataset with %d samples", len(dataset))

    sample_indices = []
    if args.sample_indices.strip():
        sample_indices = [
            int(idx.strip())
            for idx in args.sample_indices.split(",")
            if idx.strip()
        ]

    run_decompilation(
        dataset,
        config,
        sample_indices=sample_indices,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
