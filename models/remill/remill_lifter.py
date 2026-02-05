"""
Remill-based lifting: assembly -> bytes -> remill-lift -> LLVM IR -> assembly.

Uses remill-lift to translate x86 assembly machine code to LLVM IR,
then compiles the IR (with Remill runtime) to assembly.
"""

import os
import subprocess
import tempfile
import logging
from typing import Tuple, Optional

from utils.preprocessing_assembly import preprocessing_assembly

logging.basicConfig(format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s ", level=logging.INFO)


# Default Remill installation path
DEFAULT_REMILL_BUILD = os.path.expanduser("~/Software/remill/build")
REMILL_LIFT = "remill-lift-17"
REMILL_LLVM_LINK = "remill-llvm-link-17"
REMILL_RUNTIME_AMD64 = "lib/Arch/X86/Runtime/amd64.bc"


def _run_cmd(cmd: list, timeout: int = 60, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    return result.returncode, result.stdout, result.stderr


def assembly_to_bytes(assembly: str, work_dir: str, clang_path: str = "clang") -> Optional[str]:
    """
    Assemble assembly text to object file, then extract raw bytes from .text section.
    Returns hex-encoded byte string for remill-lift, or None on failure.
    """
    asm_path = os.path.join(work_dir, "input.s")
    obj_path = os.path.join(work_dir, "input.o")
    bin_path = os.path.join(work_dir, "input.bin")

    with open(asm_path, "w") as f:
        f.write(assembly)

    # Assemble to object file
    cmd = [clang_path, "-c", "-target", "x86_64-unknown-linux-gnu", asm_path, "-o", obj_path]
    ret, stdout, stderr = _run_cmd(cmd, timeout=30)
    if ret != 0:
        logging.error(f"Assembly failed: {stderr}")
        return None

    # Extract .text section to binary
    cmd = ["objcopy", "-O", "binary", "--only-section=.text", obj_path, bin_path]
    ret, stdout, stderr = _run_cmd(cmd, timeout=10)
    if ret != 0:
        logging.error(f"objcopy failed: {stderr}")
        return None

    if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
        logging.error("Empty or missing .text section")
        return None

    # Convert to hex
    with open(bin_path, "rb") as f:
        raw = f.read()
    return raw.hex()


def remill_lift(
    hex_bytes: str,
    work_dir: str,
    remill_build: str,
    arch: str = "amd64",
) -> Tuple[bool, Optional[str], str]:
    """
    Run remill-lift on hex-encoded bytes.
    Returns (success, path_to_ir_file, error_msg).
    """
    ir_path = os.path.join(work_dir, "lifted.ll")
    lift_bin = os.path.join(remill_build, "bin", "lift", f"remill-lift-17")
    if not os.path.exists(lift_bin):
        lift_bin = os.path.join(remill_build, "remill-lift-17")
    if not os.path.exists(lift_bin):
        return False, None, f"remill-lift not found at {lift_bin}"

    cmd = [
        lift_bin,
        "--arch", arch,
        "--ir_out", ir_path,
        "--bytes", hex_bytes,
    ]
    ret, stdout, stderr = _run_cmd(cmd, timeout=60)
    if ret != 0:
        return False, None, stderr or stdout
    if not os.path.exists(ir_path):
        return False, None, "remill-lift produced no output"
    return True, ir_path, ""


def link_with_runtime(
    ir_path: str,
    work_dir: str,
    remill_build: str,
) -> Tuple[bool, Optional[str], str]:
    """
    Link lifted IR with Remill amd64 runtime using llvm-link.
    Returns (success, path_to_linked_bc, error_msg).
    """
    linked_path = os.path.join(work_dir, "linked.bc")
    runtime_path = os.path.join(remill_build, REMILL_RUNTIME_AMD64)
    link_bin = os.path.join(remill_build, "remill-llvm-link-17")

    if not os.path.exists(runtime_path):
        return False, None, f"Remill runtime not found: {runtime_path}"
    if not os.path.exists(link_bin):
        return False, None, f"remill-llvm-link not found: {link_bin}"

    cmd = [link_bin, ir_path, runtime_path, "-o", linked_path]
    ret, stdout, stderr = _run_cmd(cmd, timeout=60)
    if ret != 0:
        return False, None, stderr or stdout
    if not os.path.exists(linked_path):
        return False, None, "llvm-link produced no output"
    return True, linked_path, ""


def bc_to_assembly(bc_path: str, work_dir: str, llc_path: str = "llc") -> Tuple[bool, Optional[str], str]:
    """
    Compile LLVM bitcode to assembly using llc.
    Returns (success, path_to_assembly, error_msg).
    """
    asm_path = os.path.join(work_dir, "output.s")
    cmd = [llc_path, bc_path, "-o", asm_path]
    ret, stdout, stderr = _run_cmd(cmd, timeout=60)
    if ret != 0:
        return False, None, stderr or stdout
    if not os.path.exists(asm_path):
        return False, None, "llc produced no output"
    return True, asm_path, ""


def lift_assembly_to_llvm_ir(
    assembly: str,
    output_dir: str,
    remill_build: str = DEFAULT_REMILL_BUILD,
    remove_comments: bool = True,
) -> Tuple[bool, Optional[str], Optional[str], str]:
    """
    Full pipeline: assembly -> bytes -> remill-lift -> link with runtime -> assembly.

    Returns:
        (lift_success, llvm_ir_content, assembly_content, error_msg)
        - lift_success: True if we produced compilable IR+assembly
        - llvm_ir_content: The lifted LLVM IR string if successful
        - assembly_content: The compiled assembly string if successful
        - error_msg: Error description on failure
    """
    os.makedirs(output_dir, exist_ok=True)
    asm_clean = preprocessing_assembly(assembly, remove_comments=remove_comments)

    # 1. Assembly -> bytes
    hex_bytes = assembly_to_bytes(asm_clean, output_dir)
    if hex_bytes is None:
        return False, None, None, "Failed to assemble or extract bytes"

    # 2. remill-lift
    ok, ir_path, err = remill_lift(hex_bytes, output_dir, remill_build)
    if not ok:
        return False, None, None, f"remill-lift failed: {err}"

    with open(ir_path, "r") as f:
        llvm_ir = f.read()

    # 3. Link with runtime
    ok, bc_path, err = link_with_runtime(ir_path, output_dir, remill_build)
    if not ok:
        return False, llvm_ir, None, f"llvm-link failed: {err}"

    # 4. Compile to assembly
    ok, asm_path, err = bc_to_assembly(bc_path, output_dir)
    if not ok:
        return False, llvm_ir, None, f"llc failed: {err}"

    with open(asm_path, "r") as f:
        asm_output = f.read()

    return True, llvm_ir, asm_output, ""
