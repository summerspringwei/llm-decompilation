"""Export Ghidra C decompilations for ExeBench assembly samples.

This script does not call any LLM API.  It compiles each dataset assembly
snippet to an object file, runs Ghidra headless, extracts the decompiled C
function, and writes ``predictions/sample_<idx>.c`` plus raw logs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from config import GhidraConfig, HOME_DIR
from utils.subprocess_utils import run_command


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_164",
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_without_loops_164",
    ),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Ghidra C predictions from dataset assembly."
    )
    parser.add_argument("--dataset_name", default="sampled_dataset_with_loops_164")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--ghidra_home", default="")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def default_output_dir(dataset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        HOME_DIR,
        "Projects",
        "validation",
        "codex",
        f"{timestamp}_{dataset_name}_ghidra-c-no-api",
    )


def selected_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices.strip():
        return [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    if args.max_samples > 0:
        return list(range(min(args.max_samples, dataset_len)))
    return list(range(dataset_len))


def extract_c(stdout: str) -> str:
    match = re.search(r"```C\n(.*?)```", stdout, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip() + "\n"


def decompile_one(record: dict, idx: int, output_dir: Path, clang: str, ghidra: GhidraConfig) -> bool:
    sample_dir = output_dir / f"sample_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        asm_path = Path(tmp_dir) / f"{record['fname']}.s"
        obj_path = Path(tmp_dir) / f"{record['fname']}.o"
        asm_path.write_text(record["asm"]["code"][-1])
        ret = subprocess.run(
            [clang, "-c", str(asm_path), "-o", str(obj_path)],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            (sample_dir / "assembly_to_object.error").write_text(ret.stderr)
            return False

        project_dir = Path(tmp_dir) / f"ghidra_project_{idx}"
        project_dir.mkdir(parents=True, exist_ok=True)
        project_name = f"project_{idx}_{record['fname']}"
        script_path = "models/ghidra_decompile/ghidra_decompile_script.py"
        cmd = [
            ghidra.headless_analyzer_path,
            str(project_dir),
            project_name,
            "-import",
            str(obj_path),
            "-overwrite",
            "-postscript",
            script_path,
            record["fname"],
        ]
        retcode, stdout, stderr = run_command(cmd, timeout=ghidra.command_timeout)

    (sample_dir / "ghidra.stdout").write_text(stdout)
    (sample_dir / "ghidra.stderr").write_text(stderr)
    if retcode != 0:
        (sample_dir / "ghidra.error").write_text(stderr)
        return False

    c_code = extract_c(stdout)
    if not c_code:
        (sample_dir / "ghidra.error").write_text("Could not extract C block from Ghidra output.")
        return False

    (sample_dir / "response.txt").write_text(c_code)
    (predictions_dir / f"sample_{idx}.c").write_text(c_code)
    return True


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(DATASET_PATHS[args.dataset_name])
    output_dir = Path(args.output_dir or default_output_dir(args.dataset_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    ghidra = GhidraConfig(
        home_dir=args.ghidra_home or GhidraConfig().home_dir,
        command_timeout=args.timeout,
    )

    ok_count = 0
    indices = selected_indices(len(dataset), args)
    for idx in tqdm(indices, desc="Exporting Ghidra C", unit="sample"):
        ok_count += int(decompile_one(dataset[idx], idx, output_dir, args.clang, ghidra))

    summary = f"decompile_success: {ok_count}\ntotal: {len(indices)}\n"
    (output_dir / "export_summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
