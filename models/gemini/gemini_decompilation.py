"""Batch decompilation entry-point.

Loads a dataset, dispatches samples to worker processes, and collects
results using the :class:`LLMDecompileRecord` pipeline.

Usage::

    python -m models.gemini.gemini_decompilation --model gpt-oss-20b --dataset_name sampled_dataset_with_loops_and_only_one_bb_164
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
from qdrant_client import QdrantClient
from tqdm import tqdm

from config import DecompilationConfig, HOME_DIR
from models.gemini.llm_decompiler import LLMDecompileRecord
from models.rag.exebench_qdrant_base import ExebenchQdrantSearch
from utils.llm_client import create_llm_client
from utils.logging_config import get_logger, setup_logging
from utils.prompt_type import PromptType

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-based decompilation on an ExeBench dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sampled_dataset_with_loops_and_only_one_bb_164",
        help="Dataset name (must match a key in DATASET_PAIRS).",
    )
    parser.add_argument("--model", type=str, default="gpt-oss-20b")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="9001")
    parser.add_argument("--qdrant_host", type=str, default="localhost")
    parser.add_argument("--qdrant_port", type=str, default="6333")
    parser.add_argument(
        "--embedding_url",
        type=str,
        default="http://localhost:8123/embed/batch",
    )
    parser.add_argument(
        "--prompt-type",
        dest="prompt_type",
        type=str,
        default="in-context-learning",
    )
    parser.add_argument(
        "--collection_name_with_idx",
        type=str,
        default="train_synth_rich_io_filtered_{idx}_preprocessed_hermessim",
    )
    parser.add_argument("--num_generate", type=int, default=8)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--use_pcode", action="store_true")
    parser.add_argument("--use_angr_trace", action="store_true")
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated original dataset indices to run, e.g. 65,99.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Run only the first N samples from the selected dataset.",
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

# Module-level references set by ``main`` before spawning workers.
_client = None
_model_name = None
_config: DecompilationConfig | None = None
_rag_search: ExebenchQdrantSearch | None = None


@contextlib.contextmanager
def _redirect_worker_output(log_path: str):
    """Redirect this worker's stdout/stderr file descriptors to *log_path*."""
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
    """Worker function executed in each subprocess."""
    faulthandler.register(signal.SIGUSR1)

    sample_dir = os.path.join(_config.output_dir, f"sample_{idx}")
    log_path = os.path.join(sample_dir, "subprocess.log")
    with _redirect_worker_output(log_path):
        logger.info("Starting decompilation worker for sample %d", idx)
        llm_record = LLMDecompileRecord(
            record=record,
            idx=idx,
            config=_config,
            llm_client=_client,
            model_name=_model_name,
            rag_search=_rag_search,
        )
        llm_record.get_initial_prompt()
        llm_record.decompile_and_evaluate(llm_record.initial_prompt, -1)
        llm_record.correct_one()
        llm_record.finalize()
        logger.info("Finished decompilation worker for sample %d", idx)
        return llm_record


def _decompile_func_from_args(args) -> tuple[int, LLMDecompileRecord]:
    """Worker wrapper for iterator-based Pool APIs."""
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
    """Run decompilation on *dataset* using *config*."""
    output_dir = config.output_dir
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    os.makedirs(os.path.join(output_dir, "similar_records"), exist_ok=True)

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

    # Persist results.
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Summary.
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
# Dataset mappings (kept near the entry-point, not in config.py, since
# they depend on CLI arguments that may vary per experiment).
# ---------------------------------------------------------------------------


def _build_dataset_pairs(
    model: str,
    num_generate: int,
    use_pcode: bool,
    remove_comments: bool,
    prompt_type: PromptType,
    use_angr_trace: bool,
) -> dict[str, tuple[str, str]]:
    """Return ``{dataset_name: (dataset_path, output_dir)}``."""
    with_comments = "without" if remove_comments else "with"
    input_label = "ghidra-pcode" if use_pcode else "assembly"
    angr_trace_label = "angr-trace" if use_angr_trace else "no-angr-trace"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def _output_dir(subset_label: str) -> str:
        return os.path.join(
            HOME_DIR,
            "Projects",
            "validation",
            model,
            (
                f"{run_timestamp}_{subset_label}_{model}-n{num_generate}-{input_label}"
                f"-{with_comments}-comments-{prompt_type}-similar-hermes-{angr_trace_label}"
            ),
        )

    return {
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
        "sampled_dataset_with_loops_164": (
            os.path.join(
                HOME_DIR,
                "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
            ),
            _output_dir("sample_loops"),
        ),
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    global _client, _model_name, _config, _rag_search

    setup_logging()
    args = parse_args()
    config = DecompilationConfig.from_args(args)
    prompt_type = PromptType(config.prompt_type)

    # Build client.
    _client, _model_name = create_llm_client(config)

    # Build RAG search.
    qdrant_client = QdrantClient(
        host=config.rag.qdrant_host, port=config.rag.qdrant_port
    )
    _rag_search = ExebenchQdrantSearch(
        config.rag.dataset_dir,
        qdrant_client,
        config.rag.embedding_url,
        config.rag.collection_name_template,
    )

    # Resolve dataset.
    dataset_pairs = _build_dataset_pairs(
        model=config.model_name,
        num_generate=config.num_generate,
        use_pcode=config.use_pcode,
        remove_comments=config.remove_comments,
        prompt_type=prompt_type,
        use_angr_trace=config.use_angr_trace,
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

    dataset = load_from_disk(dataset_path)
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
