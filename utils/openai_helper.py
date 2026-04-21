"""Backward-compatibility shim.

All functionality has been moved to:
- :mod:`utils.llm_response_parser` — code extraction and response parsing
- :mod:`utils.prompt_builder` — prompt construction
- :mod:`utils.prompt_type` — ``PromptType`` enum

This module re-exports the old public names so existing ``from utils.openai_helper import …``
statements continue to work.
"""

# Re-export PromptType
from utils.prompt_type import PromptType  # noqa: F401

# Re-export response parsing
from utils.llm_response_parser import (  # noqa: F401
    extract_llvm_code,
    extract_llvm_code_from_response,
    extract_llvm_code_from_response_text,
)

# Re-export prompt builders under the old names
from utils.prompt_builder import (  # noqa: F401
    build_compile_error_prompt as format_compile_error_prompt,
    build_execution_error_prompt as format_execution_error_prompt,
    build_execution_error_prompt_with_angr_trace as format_execution_error_prompt_with_angr_debug_trace,
    build_execution_error_prompt_with_ghidra_decompile as format_execution_error_prompt_with_ghidra_decompile_predict,
    build_llm_fix_prompt as format_llm_fix_prompt,
)

# Backward compat: the old typo-name
extrac_llvm_code_from_response_text = extract_llvm_code_from_response_text
