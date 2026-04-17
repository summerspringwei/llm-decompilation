import re
import logging
from typing import List 
from enum import Enum
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.prompt_templates import (
    BASIC_DECOMPILE_TEMPLATE,
    GENERAL_INIT_PROMPT,
    GHIDRA_PCODE_INIT_PROMPT,
    LLM4COMPILE_PROMPT,
    SIMILAR_RECORD_PROMPT,
    GHIDRA_PCODE_SIMILAR_RECORD_PROMPT,
    COMPILE_ERROR_TEMPLATE,
    TEST_ERROR_TEMPLATE,
    TEST_ERROR_TEMPLATE_WITH_GHIDRA_DECOMPILE_PREDICT,
    GHIDRA_DECOMPILE_TEMPLATE,
    LLM_FIX_PROMPT,
    TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE
)

def format_compile_error_prompt(first_prompt, predict, error_msg):
    prompt = first_prompt + COMPILE_ERROR_TEMPLATE.format(
        predict=predict,
        error_msg=error_msg
    )
    return prompt


def format_execution_error_prompt(first_prompt, predict, predict_assembly):
    prompt = first_prompt + TEST_ERROR_TEMPLATE.format(
        predict=predict,
        predict_assembly=predict_assembly
    )

    return prompt


def format_execution_error_prompt_with_ghidra_decompile_predict(first_prompt, predict, predict_ghidra_c_code):
    prompt = first_prompt + TEST_ERROR_TEMPLATE_WITH_GHIDRA_DECOMPILE_PREDICT.format(
        predict=predict,
        predict_ghidra_c_code=predict_ghidra_c_code
    )
    return prompt


def format_execution_error_prompt_with_angr_debug_trace(first_prompt, predict_llvm_ir, predict_assembly, target_execution_trace, predict_execution_trace):
    prompt = first_prompt + TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE.format(
        predict_llvm_ir=predict_llvm_ir,
        predict_assembly=predict_assembly,
        target_execution_trace=target_execution_trace,
        predict_execution_trace=predict_execution_trace
    )
    return prompt


def format_llm_fix_prompt(first_prompt, predict_llvm_ir, predict_assembly, analysis):
    prompt = first_prompt + LLM_FIX_PROMPT.format(
        predict_llvm_ir=predict_llvm_ir,
        predict_assembly=predict_assembly,
        analysis=analysis
    )
    return prompt

def extract_llvm_code(markdown_content: str):
    llvm_code_blocks = []
    # Use a non-greedy regex to match multiple code blocks
    pattern = r"```llvm\n(.*?)\n```"  # The \n is crucial to prevent matching across blocks
    matches = re.findall(pattern, markdown_content, re.DOTALL) # re.DOTALL to match across multiple lines
    if matches:
        llvm_code_blocks = matches
        return llvm_code_blocks
    # Extract code between <code> and </code
    pattern = r"<code>(.*?)</code>"
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    if matches:
        llvm_code_blocks = matches
    return llvm_code_blocks


def extrac_llvm_code_from_response_text(result: str)-> str:
    if result.find("</think>") > -1:
        result = result.split("</think>")[-1].strip()
    llvm_code = extract_llvm_code(result)
    if isinstance(llvm_code, list) and len(llvm_code) > 0:
        llvm_code = llvm_code[0]
    if len(llvm_code) == 0:
        logger.warning(f"No LLVM code found in the response: {result}")
    return llvm_code


def extract_llvm_code_from_response(response)->List[str]:
    predict_llvm_code_list = []
    if response.choices and len(response.choices) > 0:
        for choice in response.choices:
            result = choice.message.content
            # extract the content after </think>
            if result is None:
                logger.info(f"No result found in the response")
                predict_llvm_code_list.append("")
                continue
            llvm_code = extrac_llvm_code_from_response_text(result)
            predict_llvm_code_list.append(llvm_code)
    else:
        logger.warning("No choices found in the response.")
    return predict_llvm_code_list



class PromptType(Enum):
    BASIC = "basic"
    SIMILAR_RECORD = "in-context-learning"
    GHIDRA_DECOMPILE = "ghidra-decompile"
    GHIDRA_PCODE_DECOMPILE = "ghidra-pcode-decompile"
    GHIDRA_DECOMPILE_WITH_PREDICT = "ghidra-decompile-with-predict"
    COMPILE_FIX = "compile-fix"
    LLM_FIX = "llm-fix"
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
