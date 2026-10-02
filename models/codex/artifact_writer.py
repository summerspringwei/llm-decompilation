"""Write Codex-provided LLVM IR artifacts into the LLMDecompileRecord flow.

This module intentionally does not create an LLM client and does not call any
external model API.  It only wraps supplied LLVM IR strings in the minimal
response shape expected by ``utils.llm_response_parser`` and then delegates
compilation/execution scoring to ``LLMDecompileRecord``.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from types import MethodType
from typing import Iterable, Optional


@dataclass
class CodexMessage:
    """Minimal assistant message compatible with the response parser."""

    content: Optional[str]
    role: str = "assistant"


@dataclass
class CodexChoice:
    """Minimal choice object compatible with OpenAI-style responses."""

    message: CodexMessage
    index: int
    finish_reason: str = "stop"


@dataclass
class CodexCompletion:
    """Picklable response container with ``choices[*].message.content``."""

    choices: list[CodexChoice]
    model: str = "codex-artifact"
    id: str = "codex-artifact"


def _as_llvm_markdown(llvm_ir: str) -> str:
    """Wrap raw LLVM IR in a fenced block if the caller did not already do so."""
    if "```llvm" in llvm_ir:
        return llvm_ir
    return f"```llvm\n{llvm_ir.rstrip()}\n```"


def build_response(
    llvm_ir_strings: Iterable[str],
    model: str = "codex-artifact",
) -> CodexCompletion:
    """Build the response object consumed by ``LLMDecompileRecord``.

    Args:
        llvm_ir_strings: One or more LLVM IR predictions.
        model: Label stored in the pickled response object.
    """
    choices = [
        CodexChoice(
            message=CodexMessage(content=_as_llvm_markdown(llvm_ir)),
            index=idx,
        )
        for idx, llvm_ir in enumerate(llvm_ir_strings)
    ]
    return CodexCompletion(choices=choices, model=model, id=f"{model}-local")


def evaluate_llvm_ir_strings(
    record,
    idx: int,
    llvm_ir_strings: Iterable[str],
    output_dir: str,
    *,
    config=None,
    prompt: Optional[str] = None,
    retry_count: int = -1,
    model_name: str = "codex-artifact",
    overwrite_response: bool = False,
    save_results: bool = True,
):
    """Evaluate supplied LLVM IR strings through ``LLMDecompileRecord``.

    The function writes the same prompt/response/sample artifacts used by the
    normal pipeline, but the response comes from ``llvm_ir_strings`` rather than
    an API call.
    """
    from config import DecompilationConfig
    from models.gemini.llm_decompiler import LLMDecompileRecord

    predictions = list(llvm_ir_strings)
    if not predictions:
        raise ValueError("llvm_ir_strings must contain at least one prediction")

    os.makedirs(output_dir, exist_ok=True)
    if config is None:
        config = DecompilationConfig()
    config.output_dir = output_dir
    config.num_generate = len(predictions)

    response = build_response(predictions, model=model_name)
    prompt_text = prompt or (
        "Codex-provided LLVM IR artifact. No external LLM API was called."
    )

    llm_record = LLMDecompileRecord(
        record=record,
        idx=idx,
        config=config,
        llm_client=None,
        model_name=model_name,
        rag_search=None,
    )
    llm_record.initial_prompt = prompt_text

    def _save_and_load_response(self, response_path: str, prompt: str):
        if os.path.exists(response_path) and not overwrite_response:
            with open(response_path, "rb") as f:
                return pickle.load(f)
        with open(response_path, "wb") as f:
            pickle.dump(response, f)
        return response

    llm_record._save_and_load_response = MethodType(  # type: ignore[method-assign]
        _save_and_load_response,
        llm_record,
    )
    try:
        llm_record.decompile_and_evaluate(prompt_text, retry_count)
    finally:
        if "_save_and_load_response" in llm_record.__dict__:
            del llm_record.__dict__["_save_and_load_response"]
        llm_record.finalize()

    if save_results:
        with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
            pickle.dump([llm_record], f)

    return llm_record


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local Codex LLVM IR artifacts without an LLM API."
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument(
        "--llvm_ir_file",
        action="append",
        required=True,
        help="Path to a predicted LLVM IR file. Repeat for multiple choices.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--model_name", default="codex-artifact")
    parser.add_argument("--overwrite_response", action="store_true")
    parser.add_argument("--no_results_pkl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from datasets import load_from_disk
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency for CLI dataset loading: {exc}") from exc

    dataset = load_from_disk(args.dataset_path)
    prompt = args.prompt
    if args.prompt_file:
        prompt = _read_text(args.prompt_file)
    predictions = [_read_text(path) for path in args.llvm_ir_file]

    evaluate_llvm_ir_strings(
        record=dataset[args.idx],
        idx=args.idx,
        llvm_ir_strings=predictions,
        output_dir=args.output_dir,
        prompt=prompt or None,
        model_name=args.model_name,
        overwrite_response=args.overwrite_response,
        save_results=not args.no_results_pkl,
    )


if __name__ == "__main__":
    main()
