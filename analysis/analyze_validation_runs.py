"""Compare validation runs and inspect angr-trace retry prompts.

The script intentionally uses the lightweight result unpickler from
``models.gemini.summarize_validation`` so it can read validation pickles even
when the full decompilation runtime dependencies are not installed.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

from models.gemini.summarize_validation import (
    ResultUnpickler,
    record_prediction_success,
    record_target_success,
)


TRACE_MARKER = "execution trace of the ground truth assembly code"
PROMPT_RE = re.compile(r"prompt_(?P<idx>\d+)(?:_retry_(?P<retry>-?\d+))?\.pkl$")
CODE_BLOCK_RE = re.compile(r"```assembly\n(.*?)\n```", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and compare decompilation validation runs."
    )
    parser.add_argument("--angr_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max_failed", type=int, default=50)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return ResultUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)


def load_results(run_dir: Path) -> list[Any]:
    results_path = run_dir / "results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = load_pickle(results_path)
    if not isinstance(results, list):
        raise TypeError(f"{results_path} contains {type(results).__name__}")
    return results


def record_idx(record: Any, fallback: int) -> int:
    return int(getattr(record, "idx", fallback))


def retry_validations(record: Any) -> dict[int, Any]:
    value = getattr(record, "retry_response_validation", {})
    return value if isinstance(value, dict) else {}


def validation_success(validation: Any) -> tuple[bool, bool, int, int]:
    results = getattr(validation, "predict_evaluation_results_list", []) or []
    compile_count = sum(bool(getattr(r, "compile_success", False)) for r in results)
    exec_count = sum(bool(getattr(r, "execution_success", False)) for r in results)
    return compile_count > 0, exec_count > 0, compile_count, exec_count


def retry_summary(record: Any) -> list[dict[str, Any]]:
    rows = []
    for retry, validation in sorted(retry_validations(record).items()):
        compile_ok, exec_ok, compile_count, exec_count = validation_success(validation)
        rows.append(
            {
                "retry": retry,
                "compile_ok": compile_ok,
                "exec_ok": exec_ok,
                "compile_count": compile_count,
                "exec_count": exec_count,
            }
        )
    return rows


def first_success_retry(record: Any) -> int | None:
    for row in retry_summary(record):
        if row["exec_ok"]:
            return int(row["retry"])
    return None


def build_index(results: list[Any]) -> dict[int, Any]:
    return {record_idx(record, i): record for i, record in enumerate(results)}


def compare_runs(angr_results: list[Any], baseline_results: list[Any] | None) -> list[str]:
    lines: list[str] = []
    angr_by_idx = build_index(angr_results)
    baseline_by_idx = build_index(baseline_results) if baseline_results else {}

    angr_exec = {
        idx for idx, record in angr_by_idx.items() if record_prediction_success(record)[1]
    }
    angr_compile = {
        idx for idx, record in angr_by_idx.items() if record_prediction_success(record)[0]
    }
    angr_target_exec_fail = {
        idx for idx, record in angr_by_idx.items() if not record_target_success(record)[1]
    }
    baseline_exec = {
        idx
        for idx, record in baseline_by_idx.items()
        if record_prediction_success(record)[1]
    }
    baseline_compile = {
        idx
        for idx, record in baseline_by_idx.items()
        if record_prediction_success(record)[0]
    }
    baseline_target_exec_fail = {
        idx
        for idx, record in baseline_by_idx.items()
        if not record_target_success(record)[1]
    }

    lines.append("## Run Comparison")
    lines.append("")
    lines.append(f"angr compile successes: {len(angr_compile)}")
    lines.append(f"angr execution successes: {len(angr_exec)}")
    lines.append(f"angr target execution failures: {sorted(angr_target_exec_fail)}")
    if baseline_results is not None:
        lines.append(f"baseline compile successes: {len(baseline_compile)}")
        lines.append(f"baseline execution successes: {len(baseline_exec)}")
        lines.append(
            f"baseline target execution failures: {sorted(baseline_target_exec_fail)}"
        )
        lines.append(f"fixed only by angr: {sorted(angr_exec - baseline_exec)}")
        lines.append(f"regressed with angr: {sorted(baseline_exec - angr_exec)}")
        lines.append(f"failed in both: {sorted(set(angr_by_idx) - (angr_exec | baseline_exec))}")
    lines.append("")
    return lines


def prompt_stats(run_dir: Path) -> tuple[list[str], dict[tuple[int, int], dict[str, Any]]]:
    lines: list[str] = []
    per_prompt: dict[tuple[int, int], dict[str, Any]] = {}
    lengths = []
    trace_lengths = []
    empty_trace_prompts = 0
    trace_prompt_count = 0

    for path in sorted(run_dir.glob("prompt_*.pkl")):
        match = PROMPT_RE.match(path.name)
        if not match:
            continue
        retry_raw = match.group("retry")
        if retry_raw is None:
            continue

        prompt = load_pickle(path)
        if not isinstance(prompt, str):
            continue

        idx = int(match.group("idx"))
        retry = int(retry_raw)
        has_trace = TRACE_MARKER in prompt
        lengths.append(len(prompt))
        blocks = CODE_BLOCK_RE.findall(prompt) if has_trace else []
        target_trace = blocks[-2] if len(blocks) >= 2 else ""
        predict_trace = blocks[-1] if len(blocks) >= 1 else ""
        total_trace_len = len(target_trace) + len(predict_trace)
        if has_trace:
            trace_prompt_count += 1
            trace_lengths.append(total_trace_len)
            if not target_trace.strip() or not predict_trace.strip():
                empty_trace_prompts += 1
        per_prompt[(idx, retry)] = {
            "has_trace": has_trace,
            "prompt_len": len(prompt),
            "trace_len": total_trace_len,
            "target_lines": len(target_trace.splitlines()),
            "predict_lines": len(predict_trace.splitlines()),
        }

    lines.append("## Prompt Stats")
    lines.append("")
    lines.append(f"retry prompts: {len(per_prompt)}")
    lines.append(f"angr trace prompts: {trace_prompt_count}")
    if lengths:
        lines.append(
            "prompt chars min/avg/max: "
            f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}"
        )
    if trace_lengths:
        lines.append(
            "trace chars min/avg/max: "
            f"{min(trace_lengths)}/{sum(trace_lengths)//len(trace_lengths)}/{max(trace_lengths)}"
        )
    lines.append(f"trace prompts with empty target or predict trace: {empty_trace_prompts}")
    lines.append("")
    return lines, per_prompt


def retry_effectiveness(
    angr_results: list[Any],
    prompt_info: dict[tuple[int, int], dict[str, Any]],
    max_failed: int,
) -> list[str]:
    lines: list[str] = []
    trace_retry_outcomes = Counter()
    failed_rows = []

    for fallback, record in enumerate(angr_results):
        idx = record_idx(record, fallback)
        rows = retry_summary(record)
        row_by_retry = {row["retry"]: row for row in rows}
        for (prompt_idx, retry), info in prompt_info.items():
            if prompt_idx != idx or not info["has_trace"]:
                continue
            outcome = row_by_retry.get(retry)
            if outcome is None:
                continue
            key = "exec_success" if outcome["exec_ok"] else "no_exec_success"
            trace_retry_outcomes[key] += 1

        if not record_prediction_success(record)[1]:
            failed_rows.append(
                {
                    "idx": idx,
                    "retries": rows,
                    "first_success_retry": first_success_retry(record),
                }
            )

    lines.append("## Retry Effectiveness")
    lines.append("")
    lines.append(f"trace-prompt retries that produced execution success: {trace_retry_outcomes['exec_success']}")
    lines.append(f"trace-prompt retries without execution success: {trace_retry_outcomes['no_exec_success']}")
    lines.append("")
    lines.append(f"failed indices after all retries: {[row['idx'] for row in failed_rows[:max_failed]]}")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    angr_results = load_results(args.angr_dir)
    baseline_results = load_results(args.baseline_dir) if args.baseline_dir else None

    lines: list[str] = []
    lines.extend(compare_runs(angr_results, baseline_results))
    prompt_lines, prompt_info = prompt_stats(args.angr_dir)
    lines.extend(prompt_lines)
    lines.extend(retry_effectiveness(angr_results, prompt_info, args.max_failed))

    text = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
