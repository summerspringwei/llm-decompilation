"""Codex-internal decompilation artifact/evaluation runner.

This module intentionally does not create an OpenAI client and does not call
any model API.  It mirrors the validation artifact layout used by
``models.gemini.gemini_decompilation`` while taking LLVM IR predictions from
local files produced by this Codex session or by subagents.
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from qdrant_client import QdrantClient
from tqdm import tqdm

from config import DecompilationConfig, HOME_DIR
from models.gemini.llm_decompiler import LLMDecompileRecord
from models.rag.exebench_qdrant_base import ExebenchQdrantSearch
from utils.logging_config import setup_logging
from utils.prompt_type import PromptType


DATASET_PATHS = {
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_without_loops_164",
    ),
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Codex-internal LLVM IR predictions without model API calls."
    )
    parser.add_argument("--dataset_name", default="sampled_dataset_with_loops_164")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--predictions_dir",
        default="",
        help="Directory containing sample_<idx>.ll or sample_<idx>.md predictions.",
    )
    parser.add_argument(
        "--mode",
        choices=["export-prompts", "evaluate-predictions"],
        default="export-prompts",
    )
    parser.add_argument("--prompt-type", dest="prompt_type", default="basic")
    parser.add_argument("--num_generate", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--qdrant_host", default="localhost")
    parser.add_argument("--qdrant_port", default="6333")
    parser.add_argument("--embedding_url", default="http://localhost:8123/embed/batch")
    parser.add_argument(
        "--collection_name_with_idx",
        default="train_synth_rich_io_filtered_{idx}_preprocessed_hermessim",
    )
    parser.add_argument(
        "--allow_oracle_target",
        action="store_true",
        help="Use dataset target LLVM IR as prediction; intended only for verifier smoke tests.",
    )
    return parser.parse_args()


def default_output_dir(dataset_name: str, prompt_type: str, num_generate: int) -> str:
    subset = {
        "sampled_dataset_with_loops_and_only_one_bb_164": "sample_only_one_bb",
        "sampled_dataset_without_loops_164": "sample_without_loops",
        "sampled_dataset_with_loops_164": "sample_loops",
    }[dataset_name]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        HOME_DIR,
        "Projects",
        "validation",
        "codex",
        (
            f"{timestamp}_{subset}_codex-n{num_generate}-assembly-without-comments-"
            f"{prompt_type}-no-api"
        ),
    )


def selected_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices.strip():
        return [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if args.max_samples > 0:
        return list(range(min(args.max_samples, dataset_len)))
    return list(range(dataset_len))


def maybe_rag_search(config: DecompilationConfig, prompt_type: str):
    if PromptType(prompt_type) != PromptType.SIMILAR_RECORD:
        return None
    client = QdrantClient(host=config.rag.qdrant_host, port=config.rag.qdrant_port)
    return ExebenchQdrantSearch(
        config.rag.dataset_dir,
        client,
        config.rag.embedding_url,
        config.rag.collection_name_template,
    )


def make_config(args: argparse.Namespace) -> DecompilationConfig:
    config = DecompilationConfig.from_args(args)
    config.model_name = "codex-internal-session"
    config.fix_prompt_type = "compile-fix"
    config.num_retry = 0
    config.num_processes = 1
    config.output_dir = args.output_dir or default_output_dir(
        args.dataset_name, args.prompt_type, args.num_generate
    )
    return config


def make_response(idx: int, retry_count: int, predictions: list[str]) -> ChatCompletion:
    choices = []
    for choice_idx, llvm_ir in enumerate(predictions):
        content = f"```llvm\n{llvm_ir.strip()}\n```"
        choices.append(
            Choice(
                finish_reason="stop",
                index=choice_idx,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        )
    retry_label = "initial" if retry_count == -1 else f"retry-{retry_count}"
    return ChatCompletion(
        id=f"codex-{idx}-{retry_label}",
        choices=choices,
        created=0,
        model="codex-internal-session",
        object="chat.completion",
    )


def prediction_paths(predictions_dir: Path, idx: int) -> list[Path]:
    candidates = [
        predictions_dir / f"sample_{idx}.ll",
        predictions_dir / f"sample_{idx}.md",
        predictions_dir / f"{idx}.ll",
        predictions_dir / f"{idx}.md",
    ]
    return [path for path in candidates if path.exists()]


def load_predictions(predictions_dir: Path, idx: int) -> list[str]:
    paths = prediction_paths(predictions_dir, idx)
    if not paths:
        raise FileNotFoundError(
            f"No prediction found for sample {idx} in {predictions_dir}"
        )
    return [paths[0].read_text()]


def write_prompt(record, idx: int, config: DecompilationConfig, rag_search) -> str:
    sample_dir = Path(config.output_dir) / f"sample_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    llm_record = LLMDecompileRecord(
        record=record,
        idx=idx,
        config=config,
        llm_client=None,
        model_name=config.model_name,
        rag_search=rag_search,
    )
    prompt = llm_record.get_initial_prompt()
    (sample_dir / "prompt.txt").write_text(prompt)
    with open(Path(config.output_dir) / f"prompt_{idx}.pkl", "wb") as f:
        pickle.dump(prompt, f)
    return prompt


def export_prompts(dataset, indices: list[int], config: DecompilationConfig, rag_search) -> None:
    prompts_dir = Path(config.output_dir) / "codex_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for idx in tqdm(indices, desc="Exporting prompts", unit="sample"):
        prompt = write_prompt(dataset[idx], idx, config, rag_search)
        (prompts_dir / f"sample_{idx}.md").write_text(prompt)


def evaluate_predictions(
    dataset,
    indices: list[int],
    config: DecompilationConfig,
    rag_search,
    predictions_dir: Path | None,
    allow_oracle_target: bool,
) -> list[LLMDecompileRecord]:
    results: list[LLMDecompileRecord] = []
    for idx in tqdm(indices, desc="Evaluating Codex predictions", unit="sample"):
        record = dataset[idx]
        prompt = write_prompt(record, idx, config, rag_search)
        if allow_oracle_target:
            predictions = [record["llvm_ir"]["code"][-1]]
        else:
            if predictions_dir is None:
                raise ValueError("--predictions_dir is required unless --allow_oracle_target is set")
            predictions = load_predictions(predictions_dir, idx)
        response = make_response(idx, -1, predictions)
        with open(Path(config.output_dir) / f"response_{idx}.pkl", "wb") as f:
            pickle.dump(response, f)

        llm_record = LLMDecompileRecord(
            record=record,
            idx=idx,
            config=config,
            llm_client=None,
            model_name=config.model_name,
            rag_search=rag_search,
        )
        llm_record.initial_prompt = prompt
        validation = llm_record.evaluate_response(prompt, response, -1)
        llm_record.retry_response_validation[-1] = validation
        validation.dump_correct_llvm_ir(os.path.join(config.output_dir, f"sample_{idx}"))
        llm_record.finalize()
        results.append(llm_record)

    with open(Path(config.output_dir) / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    return results


def summarize(results: list[LLMDecompileRecord]) -> str:
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
    setup_logging()
    args = parse_args()
    config = make_config(args)
    dataset_path = DATASET_PATHS[args.dataset_name]
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(dataset_path)
    indices = selected_indices(len(dataset), args)
    rag_search = maybe_rag_search(config, args.prompt_type)

    if args.mode == "export-prompts":
        export_prompts(dataset, indices, config, rag_search)
        print(f"exported {len(indices)} prompts to {config.output_dir}")
        return

    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None
    results = evaluate_predictions(
        dataset,
        indices,
        config,
        rag_search,
        predictions_dir,
        args.allow_oracle_target,
    )
    summary = summarize(results)
    (Path(config.output_dir) / "summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {config.output_dir}")


if __name__ == "__main__":
    main()
