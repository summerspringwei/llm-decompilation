import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            elif result.find("</think>") > -1:
                result = result.split("</think>")[-1].strip()
            llvm_code = extract_llvm_code(result)
            if isinstance(llvm_code, list) and len(llvm_code) > 0:
                llvm_code = llvm_code[0]
            if len(llvm_code) == 0:
                logger.warning(f"No LLVM code found in the response: {result}")
            predict_llvm_code_list.append(llvm_code)
    else:
        logger.warning("No choices found in the response.")
    return predict_llvm_code_list


BASIC_DECOMPILE_TEMPLATE ="""
        Please decompile the following assembly code to LLVM IR and please place the final generated LLVM IR code between ```llvm and ```: 
        {target_assembly}
        Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once.
        For the global variables, please declare (not define) them in the LLVM IR code.
        Set the target data layout and target triple to the following:
        ```llvm
        target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
        target triple = "x86_64-unknown-linux-gnu"
        ```
        """

COMPILE_ERROR_TEMPLATE = BASIC_DECOMPILE_TEMPLATE + """
            You generated the following LLVM IR but it is failed to be compiled: ```llvm\n{predict}```\n 
            The compilation error message is as follows: {error_msg}
            Please correct the LLVM IR code and make sure it is correct.\n
            place the final generated LLVM IR code between ```llvm and ```.
            """

TEST_ERROR_TEMPLATE = BASIC_DECOMPILE_TEMPLATE + """
        Then you generated the following LLVM IR: ```llvm\n{predict}```\n 
        After I compile the LLVM IR you provided, the generated assembly is: {predict_assembly}\n
        The result is not right. Please compare with the original result and re-generate the LLVM IR.\n
        Place the final generated LLVM IR code between ```llvm and ```.
        """


def format_decompile_prompt(target_assembly):
    prompt = BASIC_DECOMPILE_TEMPLATE.format(
        target_assembly=target_assembly
    )
    return prompt


def format_compile_error_prompt(target_assembly, predict, error_msg):
    prompt = COMPILE_ERROR_TEMPLATE.format(
        target_assembly=target_assembly,
        predict=predict,
        error_msg=error_msg
    )
    return prompt


def format_execution_error_prompt(target_assembly, predict, predict_assembly):
    prompt = TEST_ERROR_TEMPLATE.format(
        target_assembly=target_assembly,
        predict=predict,
        predict_assembly=predict_assembly
    )

    return prompt
