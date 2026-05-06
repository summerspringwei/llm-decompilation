"""Summarize decompilation validation results.

This script scans a validation directory for ``results.pkl`` files produced by
``models/gemini/gemini_decompilation.py`` and prints the same aggregate success
counts that the decompilation run logs at the end.

Example:

    python3 models/gemini/summarize_validation.py
    python3 models/gemini/summarize_validation.py --validation_dir ~/Projects/validation/gpt-oss-20b
    python3 models/gemini/summarize_validation.py --validation_dir ~/Projects/validation --csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# Make unpickling work when this file is run directly from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Summary:
    result_path: Path
    total: int
    predict_compile_success: int
    predict_execution_success: int
    target_compile_success: int
    target_execution_success: int

    @property
    def run_dir(self) -> Path:
        if self.result_path == Path("TOTAL"):
            return self.result_path
        return self.result_path.parent


class PickleObject:
    """Attribute container used when full runtime classes are unavailable."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "PickleObject":
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and not self.__dict__:
            self.value = args[0]

    def __setstate__(self, state: dict[str, Any]) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.value = state


class ResultUnpickler(pickle.Unpickler):
    """Load validation result pickles without importing optional runtime deps."""

    PLACEHOLDER_MODULE_PREFIXES = (
        "config",
        "models.",
        "openai.",
        "utils.",
    )

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" or module.startswith(self.PLACEHOLDER_MODULE_PREFIXES):
            return type(name, (PickleObject,), {})
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize results.pkl files under a validation directory."
    )
    parser.add_argument(
        "--validation_dir",
        type=Path,
        default=Path("~/Projects/validation").expanduser(),
        help="Validation directory or a single run directory containing results.pkl.",
    )
    parser.add_argument(
        "--results_name",
        type=str,
        default="results.pkl",
        help="Pickle filename to search for.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV instead of a human-readable table.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, list):
        return any(as_bool(item) for item in value)
    return bool(value)


def get_attr_or_key(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def record_prediction_success(record: Any) -> tuple[bool, bool]:
    method = getattr(record, "predict_has_compile_and_execution_success", None)
    if callable(method):
        return tuple(bool(v) for v in method())

    retry_validations = get_attr_or_key(record, "retry_response_validation")
    if isinstance(retry_validations, dict):
        compile_success = False
        execution_success = False
        for response_validation in retry_validations.values():
            eval_results = get_attr_or_key(
                response_validation, "predict_evaluation_results_list", []
            )
            for eval_result in eval_results:
                compile_success = (
                    compile_success
                    or as_bool(get_attr_or_key(eval_result, "compile_success"))
                )
                execution_success = (
                    execution_success
                    or as_bool(get_attr_or_key(eval_result, "execution_success"))
                )
        return compile_success, execution_success

    compile_success = get_attr_or_key(record, "predict_compile_success")
    execution_success = get_attr_or_key(record, "predict_execution_success")
    if compile_success is not None or execution_success is not None:
        return as_bool(compile_success), as_bool(execution_success)

    compile_success = get_attr_or_key(record, "compile_success")
    execution_success = get_attr_or_key(record, "execution_success")
    return as_bool(compile_success), as_bool(execution_success)


def record_target_success(record: Any) -> tuple[bool, bool]:
    method = getattr(record, "target_has_compile_and_execution_success", None)
    if callable(method):
        return tuple(bool(v) for v in method())

    retry_validations = get_attr_or_key(record, "retry_response_validation")
    if isinstance(retry_validations, dict):
        response_validation = retry_validations.get(-1)
        if response_validation is not None:
            eval_result = get_attr_or_key(
                response_validation, "target_evaluation_result"
            )
            return (
                as_bool(get_attr_or_key(eval_result, "compile_success")),
                as_bool(get_attr_or_key(eval_result, "execution_success")),
            )

    compile_success = get_attr_or_key(record, "target_compile_success")
    execution_success = get_attr_or_key(record, "target_execution_success")
    return as_bool(compile_success), as_bool(execution_success)


def find_result_files(validation_dir: Path, results_name: str) -> list[Path]:
    validation_dir = validation_dir.expanduser().resolve()
    if validation_dir.is_file():
        return [validation_dir]

    direct_result = validation_dir / results_name
    if direct_result.exists():
        return [direct_result]

    return sorted(validation_dir.rglob(results_name))


def load_results(path: Path) -> list[Any]:
    with path.open("rb") as f:
        results = ResultUnpickler(f).load()
    if not isinstance(results, list):
        raise TypeError(f"{path} contains {type(results).__name__}, expected list")
    return results


def summarize_file(path: Path) -> Summary:
    results = load_results(path)
    pred_compile = pred_exec = target_compile = target_exec = 0

    for record in results:
        pc, pe = record_prediction_success(record)
        tc, te = record_target_success(record)
        pred_compile += int(pc)
        pred_exec += int(pe)
        target_compile += int(tc)
        target_exec += int(te)

    return Summary(
        result_path=path,
        total=len(results),
        predict_compile_success=pred_compile,
        predict_execution_success=pred_exec,
        target_compile_success=target_compile,
        target_execution_success=target_exec,
    )


def summarize_many(paths: Iterable[Path]) -> list[Summary]:
    summaries: list[Summary] = []
    for path in paths:
        try:
            summaries.append(summarize_file(path))
        except Exception as exc:
            print(f"warning: failed to summarize {path}: {exc}", file=sys.stderr)
    return summaries


def total_summary(summaries: list[Summary]) -> Summary:
    return Summary(
        result_path=Path("TOTAL"),
        total=sum(s.total for s in summaries),
        predict_compile_success=sum(s.predict_compile_success for s in summaries),
        predict_execution_success=sum(s.predict_execution_success for s in summaries),
        target_compile_success=sum(s.target_compile_success for s in summaries),
        target_execution_success=sum(s.target_execution_success for s in summaries),
    )


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def print_table(summaries: list[Summary], validation_dir: Path) -> None:
    if not summaries:
        print(f"No results.pkl files found under {validation_dir.expanduser()}")
        return

    rows = summaries + [total_summary(summaries)]
    run_width = max(len(str(s.run_dir)) for s in rows)
    header = (
        f"{'run_dir':<{run_width}}  {'total':>5}  "
        f"{'pred_compile':>14}  {'pred_exec':>12}  "
        f"{'target_compile':>14}  {'target_exec':>12}"
    )
    print(header)
    print("-" * len(header))
    for s in rows:
        print(
            f"{str(s.run_dir):<{run_width}}  {s.total:>5}  "
            f"{s.predict_compile_success:>5} ({pct(s.predict_compile_success, s.total):>6})  "
            f"{s.predict_execution_success:>5} ({pct(s.predict_execution_success, s.total):>6})  "
            f"{s.target_compile_success:>5} ({pct(s.target_compile_success, s.total):>6})  "
            f"{s.target_execution_success:>5} ({pct(s.target_execution_success, s.total):>6})"
        )


def print_csv(summaries: list[Summary]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "run_dir",
            "total",
            "predict_compile_success",
            "predict_execution_success",
            "target_compile_success",
            "target_execution_success",
        ]
    )
    for s in summaries + [total_summary(summaries)]:
        writer.writerow(
            [
                s.run_dir,
                s.total,
                s.predict_compile_success,
                s.predict_execution_success,
                s.target_compile_success,
                s.target_execution_success,
            ]
        )


def main() -> None:
    args = parse_args()
    result_files = find_result_files(args.validation_dir, args.results_name)
    summaries = summarize_many(result_files)
    if args.csv:
        print_csv(summaries)
    else:
        print_table(summaries, args.validation_dir)


if __name__ == "__main__":
    main()
