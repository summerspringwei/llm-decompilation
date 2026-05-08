"""Analyze execution failures in validation result directories."""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.gemini.summarize_validation import (  # noqa: E402
    ResultUnpickler,
    get_attr_or_key,
    record_prediction_success,
    record_target_success,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dirs", nargs="+", type=Path)
    return parser.parse_args()


def load_results(path: Path) -> list[Any]:
    with (path / "results.pkl").open("rb") as f:
        return ResultUnpickler(f).load()


def ok_pair(record: Any, prefix: str) -> tuple[bool, bool]:
    if prefix == "predict":
        return record_prediction_success(record)
    if prefix == "target":
        return record_target_success(record)
    return False, False


def result_flags(er: Any) -> tuple[bool, bool]:
    return bool(getattr(er, "compile_success", False)), bool(
        getattr(er, "execution_success", False)
    )


def retry_items(record: Any) -> list[tuple[int, Any]]:
    retry = getattr(record, "retry_response_validation", {}) or {}
    if isinstance(retry, dict):
        return sorted((k, v) for k, v in retry.items() if k != -1)
    return list(enumerate(retry))


def best_final_result(record: Any) -> Any | None:
    best = getattr(record, "best_result", None)
    if best:
        return best
    final = getattr(record, "final_evaluation_result", None)
    if final:
        return final
    for _, rv in reversed(retry_items(record)):
        for er in get_attr_or_key(rv, "predict_evaluation_results_list", []) or []:
            if get_attr_or_key(er, "compile_success", False):
                return er
    initial = getattr(record, "initial_response_validation", None)
    if initial:
        for er in get_attr_or_key(initial, "predict_evaluation_results_list", []) or []:
            if get_attr_or_key(er, "compile_success", False):
                return er
    return None


def last_prompt_text(validation_dir: Path, idx: int, retry_no: int) -> str:
    candidates = [
        validation_dir / f"prompt_{idx}_retry_{retry_no}.pkl",
        validation_dir / f"sample_{idx}_retry_{retry_no}" / "prompt.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".pkl":
                with path.open("rb") as f:
                    return pickle.load(f)
            return path.read_text(errors="replace")
        except Exception:
            pass
    return ""


def classify_prompt(prompt: str) -> str:
    if "could not be linked into the ExeBench wrapper" in prompt:
        return "trace_link_error"
    if (
        "target executable instruction trace" in prompt
        or "execution trace of the ground truth assembly code" in prompt
    ):
        if re.search(r"; target trace lines: [1-9]", prompt):
            return "angr_trace"
        return "empty_angr_trace"
    if "compilation error message is as follows" in prompt:
        return "llc_compile_error"
    if "generated LLVM IR into the following assembly code" in prompt:
        return "execution_error"
    return "unknown"


def final_status(record: Any) -> str:
    pc, pe = ok_pair(record, "predict")
    tc, te = ok_pair(record, "target")
    if not tc or not te:
        return "target_failure"
    if pe:
        return "pass"
    if pc:
        return "predict_execution_failure"
    return "predict_compile_failure"


def record_name(record: Any) -> str:
    row = getattr(record, "record", None)
    if hasattr(row, "path"):
        return row.path
    if isinstance(row, dict):
        return row.get("path", "")
    return ""


def analyze_dir(validation_dir: Path) -> None:
    results = load_results(validation_dir)
    counters: Counter[str] = Counter()
    failure_rows: list[dict[str, Any]] = []
    failure_error_families: Counter[str] = Counter()
    failure_paths: Counter[str] = Counter()
    retry_counter: Counter[str] = Counter()
    retry_success_after_prompt: Counter[str] = Counter()

    for idx, record in enumerate(results):
        status = final_status(record)
        counters[status] += 1
        prompt_classes_seen: set[str] = set()
        for retry_no, _rv in retry_items(record):
            prompt = last_prompt_text(validation_dir, idx, retry_no)
            cls = classify_prompt(prompt)
            retry_counter[cls] += 1
            prompt_classes_seen.add(cls)
        if status == "pass":
            for cls in prompt_classes_seen:
                retry_success_after_prompt[cls] += 1
            continue

        retries = retry_items(record)
        last_retry_no = retries[-1][0] if retries else -1
        last_prompt = last_prompt_text(validation_dir, idx, last_retry_no)
        final_er = best_final_result(record)
        compile_ok, exec_ok = result_flags(final_er) if final_er else (False, False)
        err = (getattr(final_er, "error_msg", "") or "").strip() if final_er else ""
        failure_paths[record_name(record)] += 1
        for _, rv in retry_items(record):
            for er in get_attr_or_key(rv, "predict_evaluation_results_list", []) or []:
                msg = (get_attr_or_key(er, "error_msg", "") or "").lower()
                if not msg:
                    continue
                if "multiple definition of local value" in msg:
                    failure_error_families["duplicate_ssa_name"] += 1
                elif "expected '(' in constantexpr" in msg or "getelementptr" in msg:
                    failure_error_families["bad_getelementptr_or_constexpr"] += 1
                elif "floating point constant invalid" in msg:
                    failure_error_families["bad_float_literal"] += 1
                elif "constant expression type mismatch" in msg:
                    failure_error_families["bad_vector_constant"] += 1
                elif "inline asm" in msg or "asm sideeffect" in msg:
                    failure_error_families["bad_inline_asm"] += 1
                elif "undefined value" in msg:
                    failure_error_families["undefined_ssa_value"] += 1
                elif "is not a basic block" in msg or "phi nodes" in msg:
                    failure_error_families["bad_cfg_or_phi"] += 1
                elif "unknown target property" in msg:
                    failure_error_families["bad_target_attribute"] += 1
                else:
                    failure_error_families["other_llc_error"] += 1
        failure_rows.append(
            {
                "idx": idx,
                "status": status,
                "last_prompt": classify_prompt(last_prompt),
                "num_retries": len(retries),
                "compile_ok": compile_ok,
                "exec_ok": exec_ok,
                "path": record_name(record),
                "error": " ".join(err.split())[:180],
            }
        )

    print(f"\n## {validation_dir}")
    print("summary", dict(counters))
    print("retry_prompt_counts", dict(retry_counter))
    print("passed_after_prompt_type", dict(retry_success_after_prompt))
    by_prompt = Counter(row["last_prompt"] for row in failure_rows)
    by_status = Counter(row["status"] for row in failure_rows)
    print("failure_by_status", dict(by_status))
    print("failure_by_last_prompt", dict(by_prompt))
    print("failure_error_families", dict(failure_error_families))
    print("failure_paths", dict(failure_paths))
    print("failures")
    for row in failure_rows:
        print(
            f"  idx={row['idx']:3d} status={row['status']:<26} "
            f"last={row['last_prompt']:<17} retries={row['num_retries']:<2d} "
            f"compile={int(row['compile_ok'])} exec={int(row['exec_ok'])} "
            f"path={row['path']}"
        )
        if row["error"]:
            print(f"       error={row['error']}")


def main() -> None:
    for validation_dir in parse_args().validation_dirs:
        analyze_dir(validation_dir)


if __name__ == "__main__":
    main()
