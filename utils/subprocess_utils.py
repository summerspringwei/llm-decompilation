"""Thin wrapper around :func:`subprocess.run` used throughout the codebase.

Eliminates the three independent ``_run_command`` implementations that existed
in ``llm_decompiler.py``, ``ghidra_decompile_exebench.py``, etc.
"""

import subprocess
from typing import Optional, Tuple


def run_command(
    command: list[str],
    *,
    stdin: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[int, str, str]:
    """Execute *command* and return ``(returncode, stdout, stderr)``.

    Args:
        command: Argument list (no shell expansion).
        stdin: Optional string fed to the process' standard input.
        timeout: Maximum wall-clock seconds before the process is killed.

    Returns:
        A ``(returncode, stdout, stderr)`` tuple.  Both output streams are
        decoded strings (``text=True``).
    """
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        input=stdin,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr
