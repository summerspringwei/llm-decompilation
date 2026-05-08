"""Print detailed stored fields for selected validation failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.gemini.summarize_validation import (  # noqa: E402
    ResultUnpickler,
    get_attr_or_key,
    record_prediction_success,
)


def load_results(path: Path) -> list[Any]:
    with (path / "results.pkl").open("rb") as f:
        return ResultUnpickler(f).load()


def clip(value: Any, limit: int = 300) -> str:
    return str(value).replace("\n", " ")[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()

    results = load_results(args.validation_dir)
    print("DIR", args.validation_dir)
    for idx in args.indices:
        record = results[idx]
        row = get_attr_or_key(record, "record")
        print("\nIDX", idx, "pred", record_prediction_success(record))
        print("path", get_attr_or_key(row, "path", ""))
        retry = get_attr_or_key(record, "retry_response_validation", {}) or {}
        retry_keys = sorted([k for k in retry if k != -1]) if isinstance(retry, dict) else []
        print("retry_keys", retry_keys)
        for key in retry_keys[-3:]:
            rv = retry[key]
            evals = get_attr_or_key(rv, "predict_evaluation_results_list", []) or []
            flags = [
                (
                    get_attr_or_key(er, "compile_success"),
                    get_attr_or_key(er, "execution_success"),
                    clip(get_attr_or_key(er, "error_msg", ""), 120),
                )
                for er in evals
            ]
            print(" retry", key, "n", len(evals), "flags", flags)
            for er_idx, er in enumerate(evals[:2]):
                print("  er", er_idx, "fields", sorted(er.__dict__.keys()))
                for name in [
                    "error_msg",
                    "input",
                    "expected_output",
                    "observed_output",
                    "execution_error",
                    "llvm_ir",
                    "assembly",
                ]:
                    value = get_attr_or_key(er, name)
                    if value:
                        print("   ", name, clip(value))


if __name__ == "__main__":
    main()
