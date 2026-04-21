"""Prompt construction for the LLM decompilation pipeline.

Every prompt-building function that was previously scattered across
``LLMDecompileRecord``, ``openai_helper``, and ``iterative_decompile`` is
consolidated here.  The functions are stateless — they accept preprocessed
inputs and return prompt strings.
"""

from __future__ import annotations

from typing import List, Optional

from utils.prompt_templates import (
    COMPILE_ERROR_TEMPLATE,
    GENERAL_INIT_PROMPT,
    GHIDRA_DECOMPILE_TEMPLATE,
    GHIDRA_PCODE_INIT_PROMPT,
    GHIDRA_PCODE_SIMILAR_RECORD_PROMPT,
    LLM_FIX_PROMPT,
    SIMILAR_RECORD_PROMPT,
    TEST_ERROR_TEMPLATE,
    TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE,
    TEST_ERROR_TEMPLATE_WITH_GHIDRA_DECOMPILE_PREDICT,
)


# ---------------------------------------------------------------------------
# Initial prompt builders
# ---------------------------------------------------------------------------


def build_basic_prompt(asm_code: str) -> str:
    """Build the basic assembly-to-LLVM-IR decompilation prompt."""
    return GENERAL_INIT_PROMPT.format(asm_code=asm_code)


def build_pcode_prompt(pcode_text: str) -> str:
    """Build a prompt using Ghidra P-code as input."""
    return GHIDRA_PCODE_INIT_PROMPT.format(p_code=pcode_text)


def build_similar_record_prompt(
    asm_code: str,
    similar_asm_code: str,
    similar_llvm_ir: str,
) -> str:
    """Build an in-context-learning prompt with a similar example."""
    return SIMILAR_RECORD_PROMPT.format(
        asm_code=asm_code,
        similar_asm_code=similar_asm_code,
        similar_llvm_ir=similar_llvm_ir,
    )


def build_pcode_similar_record_prompt(
    pcode: str,
    similar_pcode: str,
    similar_llvm_ir: str,
) -> str:
    """Build an in-context-learning prompt using Ghidra P-code."""
    return GHIDRA_PCODE_SIMILAR_RECORD_PROMPT.format(
        pcode=pcode,
        similar_pcode=similar_pcode,
        similar_llvm_ir=similar_llvm_ir,
    )


def build_ghidra_decompile_prompt(
    asm_code: str,
    ghidra_c_code: str,
) -> str:
    """Build a prompt that includes Ghidra-decompiled C code as a reference."""
    return GHIDRA_DECOMPILE_TEMPLATE.format(
        asm_code=asm_code,
        ghidra_c_code=ghidra_c_code,
    )


# ---------------------------------------------------------------------------
# Fix / retry prompt builders
# ---------------------------------------------------------------------------


def build_compile_error_prompt(
    initial_prompt: str,
    predict: str,
    error_msg: str,
) -> str:
    """Append a compile-fix instruction to *initial_prompt*."""
    return initial_prompt + COMPILE_ERROR_TEMPLATE.format(
        predict=predict,
        error_msg=error_msg,
    )


def build_execution_error_prompt(
    initial_prompt: str,
    predict: str,
    predict_assembly: str,
) -> str:
    """Append an execution-fix instruction to *initial_prompt*."""
    return initial_prompt + TEST_ERROR_TEMPLATE.format(
        predict=predict,
        predict_assembly=predict_assembly,
    )


def build_execution_error_prompt_with_ghidra_decompile(
    initial_prompt: str,
    predict: str,
    predict_ghidra_c_code: str,
) -> str:
    """Append an execution-fix instruction that includes Ghidra decompiled C."""
    return initial_prompt + TEST_ERROR_TEMPLATE_WITH_GHIDRA_DECOMPILE_PREDICT.format(
        predict=predict,
        predict_ghidra_c_code=predict_ghidra_c_code,
    )


def build_execution_error_prompt_with_angr_trace(
    initial_prompt: str,
    predict_llvm_ir: str,
    predict_assembly: str,
    target_execution_trace: str,
    predict_execution_trace: str,
) -> str:
    """Append an execution-fix instruction that includes angr traces."""
    return initial_prompt + TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE.format(
        predict_llvm_ir=predict_llvm_ir,
        predict_assembly=predict_assembly,
        target_execution_trace=target_execution_trace,
        predict_execution_trace=predict_execution_trace,
    )


def build_llm_fix_prompt(
    initial_prompt: str,
    predict_llvm_ir: str,
    predict_assembly: str,
    analysis: str,
) -> str:
    """Append an LLM-generated analysis and ask for a corrected LLVM IR."""
    return initial_prompt + LLM_FIX_PROMPT.format(
        predict_llvm_ir=predict_llvm_ir,
        predict_assembly=predict_assembly,
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Failure-analysis prompt (used by the LLM-fix flow)
# ---------------------------------------------------------------------------


def build_failure_analysis_prompt(
    target_assembly: str,
    predicted_llvm_ir: str,
    predicted_assembly: str,
    compile_error: str,
) -> str:
    """Build an expert-analysis prompt for diagnosing prediction failures.

    Used as the first step of the two-stage LLM-fix flow: this prompt asks
    the LLM to *analyse* the failure, then the analysis is fed into
    :func:`build_llm_fix_prompt` for the actual fix.
    """
    sections: List[str] = [
        (
            "You are a compiler expert. Analyze why the predicted LLVM IR "
            "fails to match the ground-truth behavior."
        ),
    ]

    if target_assembly:
        sections.append(
            "Reference Assembly Produced From Ground Truth (if available):\n\n"
            + target_assembly
        )

    sections.append("Predicted LLVM IR (failing):\n\n" + predicted_llvm_ir)

    if predicted_assembly:
        sections.append(
            "Assembly Produced From Predicted IR (if compiled):\n\n"
            + predicted_assembly
        )

    if compile_error and compile_error.strip():
        sections.append(
            "Compiler Diagnostics For Predicted IR (if it failed to compile):\n\n"
            + compile_error
        )

    sections.append(
        "Instructions:\n"
        "1) Compare reference assembly and predicted assembly semantics "
        "precisely (types, signedness, control-flow, memory operations, "
        "calling conventions, data layout, attributes, metadata).\n"
        "2) Identify exact mismatches that change observable behavior versus "
        "the assembly (e.g., incorrect GEP indices, missing 'nsw/nuw', "
        "wrong 'zext/sext', PHI node shape, loop bounds, UB triggers, "
        "aliasing, volatile/atomic, pointer casts/address spaces, "
        "byval/byref, linkage/visibility).\n"
        "3) If compilation failed, map diagnostics to the exact IR locations "
        "and propose minimal fixes.\n"
        "4) Summarize root causes and concrete corrective edits to the "
        "predicted IR.\n"
        "5) Please provide the detailed steps to fix the predicted IR."
    )

    return "\n\n".join(sections)
