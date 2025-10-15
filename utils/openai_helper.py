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





BASIC_DECOMPILE_TEMPLATE = """
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

GENERAL_INIT_PROMPT = """
Please decompile the following assembly code to LLVM IR
and please place the final generated LLVM IR code between ```llvm and ```: {asm_code} 
Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once."""

LLM4COMPILE_PROMPT = "Disassemble this code to LLVM IR: <code>{asm_code} \n</code>"

SIMILAR_RECORD_PROMPT = """
Please decompile the assembly code to LLVM IR.
Here is a example of the similar assembly code and the corresponding decompiled LLVM IR: 
```assembly
{similar_asm_code} 
```
→
```llvm
{similar_llvm_ir}
```
Please decompile the following assembly code to LLVM IR.
* Note that *
- LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once.
- The decompiled LLVM IR should be valid and can be compiled successfully.
- Place the final generated LLVM IR code between ```llvm and ```.
```assembly
{asm_code}
```
"""

COMPILE_ERROR_TEMPLATE = """
You generated the following LLVM IR but it is failed to be compiled: ```llvm
{predict}
```
The compilation error message is as follows: {error_msg}
Please correct the LLVM IR code based on the similar example and make sure it meets both the semantic and the syntax.
place the final generated LLVM IR code between ```llvm and ```.
"""

TEST_ERROR_TEMPLATE = """
Then you generated the following LLVM IR: ```llvm\n{predict}```\n 
After I compile the LLVM IR you provided, the generated assembly is: {predict_assembly}\n
The result is not right. Please compare the generated assembly with the original assembly and re-generate the LLVM IR.\n
Place the final generated LLVM IR code between ```llvm and ```.
"""


def format_decompile_prompt(target_assembly):
    prompt = BASIC_DECOMPILE_TEMPLATE.format(
        target_assembly=target_assembly
    )
    return prompt


def format_compile_error_prompt(target_assembly, predict, error_msg, in_context_learning, similar_asm_code=None, similar_llvm_ir=None):
    if in_context_learning:
        first_prompt = SIMILAR_RECORD_PROMPT.format(
            similar_asm_code=similar_asm_code,
            similar_llvm_ir=similar_llvm_ir,
            asm_code=target_assembly
        )
    else:
        first_prompt = BASIC_DECOMPILE_TEMPLATE.format(
            target_assembly=target_assembly
        )
    prompt = first_prompt + COMPILE_ERROR_TEMPLATE.format(
        target_assembly=target_assembly,
        predict=predict,
        error_msg=error_msg
    )
    return prompt


def format_execution_error_prompt(target_assembly, predict, predict_assembly, in_context_learning, similar_asm_code=None, similar_llvm_ir=None):
    if in_context_learning:
        first_prompt = SIMILAR_RECORD_PROMPT.format(
            similar_asm_code=similar_asm_code,
            similar_llvm_ir=similar_llvm_ir,
            asm_code=target_assembly
        )
    else:
        first_prompt = BASIC_DECOMPILE_TEMPLATE.format(
            target_assembly=target_assembly
        )
    prompt = first_prompt + TEST_ERROR_TEMPLATE.format(
        target_assembly=target_assembly,
        predict=predict,
        predict_assembly=predict_assembly
    )

    return prompt
