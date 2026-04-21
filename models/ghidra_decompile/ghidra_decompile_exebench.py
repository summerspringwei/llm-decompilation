"""Ghidra-based decompilation of ExeBench records.

Provides :class:`GhidraObjectFileDecompiler` for decompiling individual
object files and the batch helper :func:`ghidra_decompile_record`.

.. note::
   The function was previously named ``ghdria_decompile_record`` (typo).
   The corrected name ``ghidra_decompile_record`` is now canonical; the
   old name is kept as an alias for backward compatibility.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

from datasets import load_from_disk
from tqdm import tqdm

from exebench import LLVMAssembler, Wrapper

from config import GhidraConfig
from utils.logging_config import get_logger
from utils.subprocess_utils import run_command

logger = get_logger(__name__)

# Module-level config; can be overridden before calling ``main()``.
_ghidra_config = GhidraConfig()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class GhidraResult:
    """Stores the outcome of a single Ghidra decompilation run."""

    def __init__(self):
        self.ghidra_decompiled_raw_string: Optional[str] = None
        self.ghidra_decompiled_error_msg: Optional[str] = None
        self.ghidra_decompiled_error_code: Optional[int] = None
        self.ghidra_decompiled_c_code: Optional[str] = None


# ---------------------------------------------------------------------------
# Decompiler
# ---------------------------------------------------------------------------


class GhidraObjectFileDecompiler:
    """Manages Ghidra headless analysis for a single ExeBench record."""

    def __init__(
        self,
        record: Dict,
        ghidra_cfg: GhidraConfig = _ghidra_config,
    ):
        self.idx = -1
        self.record = record
        self.object_file_path: Optional[str] = None
        self.synth_wrapper = None
        self.func_name: str = record["func_info"]["functions"][0]["name"]
        self.func_def: str = record["func_def"]
        self.ghidra_result: Optional[GhidraResult] = None
        self._ghidra_cfg = ghidra_cfg

    def compile_to_object_file(
        self,
        output_dir: str,
        name_hint: str,
    ) -> str:
        """Compile the record's LLVM IR to an object file via ``llc``."""
        object_path = os.path.join(output_dir, f"{name_hint}.o")
        ll_path = os.path.join(output_dir, f"{name_hint}.ll")
        with open(ll_path, "w") as f:
            f.write(self.record["llvm_ir"]["code"][-1])

        retcode, _, stderr = run_command(
            ["llc", ll_path, "-filetype=obj", "-o", object_path],
            timeout=self._ghidra_cfg.command_timeout,
        )
        if retcode != 0:
            raise RuntimeError(
                f"Failed to compile function to object file: {stderr}"
            )
        return object_path

    def compile_to_executable(self, record: Dict) -> str:
        """Compile object to executable using ExeBench wrapper."""
        try:
            c_deps = (
                self.record["synth_deps"]
                + "\n"
                + self.record["synth_io_pairs"]["dummy_funcs"][0]
                + "\n"
            ).replace("typedef int bool;", "")
            self.synth_wrapper = Wrapper(
                c_deps=c_deps + "\n",
                func_c_signature=self.record["func_head_types"].replace(
                    "extern", ""
                ),
                func_assembly=record["asm"]["code"][-1],
                cpp_wrapper=record["synth_exe_wrapper"],
                assembler_backend=LLVMAssembler(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile function to executable: {e}"
            ) from e
        return self.synth_wrapper._compiled_exe_path

    def decompile_external_assembly(
        self,
        assembly: str,
        func_name: str,
    ) -> Optional[str]:
        """Decompile an external assembly snippet via Ghidra."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            asm_path = os.path.join(tmp_dir, f"{func_name}.s")
            obj_path = os.path.join(tmp_dir, f"{func_name}.o")
            with open(asm_path, "w") as f:
                f.write(assembly)

            retcode, _, stderr = run_command(
                ["clang", asm_path, "-o", obj_path], timeout=30
            )
            if retcode != 0:
                raise RuntimeError(
                    f"Failed to compile assembly to object file: {stderr}"
                )

            project_name = f"project_{self.idx}_predict_{func_name}"
            script_path = "models/ghidra_decompile/ghidra_decompile_script.py"
            cmd = [
                self._ghidra_cfg.headless_analyzer_path,
                tmp_dir,
                project_name,
                "-import",
                obj_path,
                "-overwrite",
                "-postscript",
                script_path,
                func_name,
            ]
            retcode, stdout, stderr = run_command(
                cmd, timeout=self._ghidra_cfg.command_timeout
            )
            if retcode != 0:
                raise RuntimeError(f"Ghidra headless analysis failed: {stderr}")
            return self._extract_c_code(stdout)

    @staticmethod
    def _extract_c_code(stdout: str) -> Optional[str]:
        """Extract C code between ``` ```C ``` and ``` ``` ``` markers."""
        match = re.search(r"```C\n(.*?)```", stdout, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # Keep the old public name as an alias.
    extract_c_code = _extract_c_code


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def ghidra_decompile_record(
    args: Tuple[int, Dict],
) -> Optional[GhidraObjectFileDecompiler]:
    """Process a single ``(idx, record)`` tuple through Ghidra.

    Returns:
        A populated :class:`GhidraObjectFileDecompiler`, or ``None`` on error.
    """
    idx, record = args
    try:
        decompiler = GhidraObjectFileDecompiler(record)
        decompiler.idx = idx

        with tempfile.TemporaryDirectory() as tmp_dir:
            obj_path = decompiler.compile_to_object_file(
                tmp_dir, decompiler.func_name
            )
            project_dir = os.path.join(
                tmp_dir, f"myghidra_{idx}_{decompiler.func_name}"
            )
            os.makedirs(project_dir, exist_ok=True)
            project_name = f"project_{idx}_{decompiler.func_name}"
            script_path = "models/ghidra_decompile/ghidra_decompile_script.py"

            cmd = [
                _ghidra_config.headless_analyzer_path,
                project_dir,
                project_name,
                "-import",
                obj_path,
                "-overwrite",
                "-postscript",
                script_path,
                decompiler.func_name,
            ]
            retcode, stdout, stderr = run_command(
                cmd, timeout=_ghidra_config.command_timeout
            )
            if retcode != 0:
                if "project exists" not in stderr:
                    raise RuntimeError(f"Ghidra headless analysis failed: {stderr}")
            else:
                result = GhidraResult()
                result.ghidra_decompiled_raw_string = stdout
                result.ghidra_decompiled_error_msg = stderr
                result.ghidra_decompiled_error_code = retcode
                result.ghidra_decompiled_c_code = (
                    GhidraObjectFileDecompiler._extract_c_code(stdout)
                )
                decompiler.ghidra_result = result

            return decompiler
    except Exception as e:
        logger.error("Error processing record %d: %s", idx, e)
        return None


# Backward-compatible alias for the old typo name.
ghdria_decompile_record = ghidra_decompile_record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(dataset_path: str, num_processes: int = 4):
    """Run Ghidra decompilation on all records in *dataset_path*."""
    dataset = load_from_disk(dataset_path)
    records_with_idx = list(enumerate(dataset))

    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(ghidra_decompile_record, records_with_idx),
                total=len(records_with_idx),
                desc="Processing records",
                unit="record",
            )
        )
    return results


if __name__ == "__main__":
    dataset_path_file_types = {
        (
            "/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164",
            "sampled_dataset_without_loops_164_ghidra_decompile",
        ),
    }
    for ds_path, file_name in dataset_path_file_types:
        results = main(ds_path, num_processes=40)
        with open(f"{file_name}.pkl", "wb") as f:
            pickle.dump(results, f)