"""Prompt type enumeration used across the decompilation pipeline.

Extracted from ``openai_helper.py`` so it can be imported without pulling in
prompt-formatting dependencies.
"""

from enum import Enum


class PromptType(Enum):
    """Strategy for constructing prompts for the LLM decompiler."""

    BASIC = "basic"
    SIMILAR_RECORD = "in-context-learning"
    GHIDRA_DECOMPILE = "ghidra-decompile"
    GHIDRA_PCODE_DECOMPILE = "ghidra-pcode-decompile"
    GHIDRA_DECOMPILE_WITH_PREDICT = "ghidra-decompile-with-predict"
    COMPILE_FIX = "compile-fix"
    LLM_FIX = "llm-fix"
    TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE = "angr-debug-trace"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value
