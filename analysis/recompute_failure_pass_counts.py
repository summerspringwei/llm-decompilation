"""Recompute pass counts for stored compilable predictions in failed samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.gemini.summarize_validation import (  # noqa: E402
    ResultUnpickler,
    get_attr_or_key,
    record_prediction_success,
    record_target_success,
)
from utils.evaluate_exebench import eval_assembly_with_details  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    args = parser.parse_args()

    with (args.validation_dir / "results.pkl").open("rb") as f:
        results = ResultUnpickler(f).load()

    print("idx,fname,best_retry,best_pass,total,compile_candidates,path")
    for idx, record in enumerate(results):
        pred_ok = record_prediction_success(record)
        target_ok = record_target_success(record)
        if pred_ok[1] and target_ok[1]:
            continue

        row = get_attr_or_key(record, "record")
        retry = get_attr_or_key(record, "retry_response_validation", {}) or {}
        best_retry = -1
        best_pass = -1
        best_total = 0
        compile_candidates = 0

        if isinstance(retry, dict):
            retry_items = sorted((k, v) for k, v in retry.items() if k != -1)
        else:
            retry_items = list(enumerate(retry))

        for retry_no, rv in retry_items:
            evals = get_attr_or_key(rv, "predict_evaluation_results_list", []) or []
            for er in evals:
                if not get_attr_or_key(er, "compile_success", False):
                    continue
                assembly = get_attr_or_key(er, "assembly", "") or ""
                if not assembly:
                    continue
                compile_candidates += 1
                details = eval_assembly_with_details(row, assembly)
                if details["pass_count"] > best_pass:
                    best_retry = retry_no
                    best_pass = details["pass_count"]
                    best_total = details["total_count"]

        print(
            f"{idx},{get_attr_or_key(row, 'fname', '')},{best_retry},"
            f"{max(best_pass, 0)},{best_total},{compile_candidates},"
            f"{get_attr_or_key(row, 'path', '')}"
        )


if __name__ == "__main__":
    main()
