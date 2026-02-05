import os
import argparse
import tempfile
import subprocess
from multiprocessing import Pool
from typing import Optional, Tuple

from datasets import load_from_disk
from tqdm import tqdm

_DEFAULT_CMD_TIMEOUT = 120


def _run_command(command,
                 stdin: Optional[str] = None,
                 timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[int, str, str]:
    output = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            input=stdin,
                            timeout=timeout)
    return output.returncode, output.stdout, output.stderr


def _get_ghidra_script_path() -> str:
    ghidra_home = os.environ.get("GHIDRA_HOME")
    if ghidra_home:
        return os.path.join(ghidra_home, "support/analyzeHeadless")
    return "/data1/xiachunwei/Software/ghidra_11.4.2_PUBLIC/support/analyzeHeadless"


def _compile_assembly_to_object(assembly: str, output_dir: str, func_name: str) -> str:
    asm_path = os.path.join(output_dir, f"{func_name}.s")
    obj_path = os.path.join(output_dir, f"{func_name}.o")
    with open(asm_path, "w") as f:
        f.write(assembly)
    cmd = ["clang", "-c", asm_path, "-o", obj_path]
    retcode, _, stderr = _run_command(cmd, timeout=30)
    if retcode != 0:
        raise RuntimeError(f"Failed to compile assembly to object file: {stderr}")
    return obj_path


def _get_func_name(record, fallback: str) -> str:
    try:
        return record["func_info"]["functions"][0]["name"]
    except Exception:
        return fallback


def _process_record(args):
    idx, record, output_dir = args
    func_name = _get_func_name(record, f"func_{idx}")
    sample_dir = os.path.join(output_dir, f"sample_{idx}")
    os.makedirs(sample_dir, exist_ok=True)
    output_pcode = os.path.join(sample_dir, "pcode.txt")
    asm_code = record["asm"]["code"][-1]
    llvm_ir = record["llvm_ir"]["code"][-1]
    with open(os.path.join(sample_dir, "original_asm.s"), "w") as f:
        f.write(asm_code)
    with open(os.path.join(sample_dir, "original_llvm_ir.ll"), "w") as f:
        f.write(llvm_ir)
    if os.path.exists(output_pcode):
        return output_pcode
    ghidra_script = os.path.join(
        os.path.dirname(__file__),
        "ghidra_pcode_script.py"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_path = _compile_assembly_to_object(asm_code, tmp_dir, func_name)
        project_dir = os.path.join(tmp_dir, f"ghidra_project_{idx}_{func_name}")
        os.makedirs(project_dir, exist_ok=True)
        project_name = f"project_{idx}_{func_name}"
        cmd = [
            _get_ghidra_script_path(),
            project_dir,
            project_name,
            "-import",
            obj_path,
            "-overwrite",
            "-postscript",
            ghidra_script,
            func_name,
            output_pcode,
        ]
        retcode, stdout, stderr = _run_command(cmd, timeout=_DEFAULT_CMD_TIMEOUT)
        with open(os.path.join(sample_dir, "ghidra_stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(sample_dir, "ghidra_stderr.txt"), "w") as f:
            f.write(stderr)
        if retcode != 0:
            raise RuntimeError(f"Ghidra failed for index {idx}: {stderr}")

    return output_pcode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lift Exebench assembly to Ghidra P-code")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the exebench dataset on disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "Projects/validation/ghidra_pcode"),
        help="Output directory for P-code JSON files",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of worker processes",
    )
    return parser.parse_args()


def main(dataset_path: str, output_dir: str, num_processes: int):
    dataset = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    args = [(idx, record, output_dir) for idx, record in enumerate(dataset)]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_record, args),
            total=len(args),
            desc="Generating P-code",
            unit="record",
        ))
    return results


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args.dataset_path, cli_args.output_dir, cli_args.num_processes)
