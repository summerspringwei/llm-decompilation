
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

GHIDRA_PCODE_INIT_PROMPT = """
Please decompile the following Ghidra decompiled high level P-code to LLVM IR
and please place the final generated LLVM IR code between ```llvm and ```: {p_code} 
Note that LLVM IR follow the Static Single Assignment format, which mean a variable can only be defined once.
"""

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

GHIDRA_PCODE_SIMILAR_RECORD_PROMPT = """
Please decompile the Ghidra decompiled high level P-code to LLVM IR.
Here is a example of the similar Ghidra decompiled high level P-code and the corresponding decompiled LLVM IR: 
```pcode
{similar_pcode}
```
→
```llvm
{similar_llvm_ir}
```
Please decompile the following Ghidra decompiled high level P-code to LLVM IR.
```pcode
{pcode}
```
Place the final generated LLVM IR code between ```llvm and ```.
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

GHIDRA_PCODE_TEST_ERROR_TEMPLATE = """
Then you generated the following LLVM IR: ```llvm\n{predict}```\n 
After I compile the LLVM IR you provided, the generated P-Code is: {predict_p_code}\n
The result is not right. Please compare the generated P-Code with the original P-Code and re-generate the LLVM IR.\n
Place the final generated LLVM IR code between ```llvm and ```.
"""


TEST_ERROR_TEMPLATE_WITH_GHIDRA_DECOMPILE_PREDICT = """
Then you generated the following LLVM IR: ```llvm\n{predict}```\n 
However your generated LLVM IR is not right. 
To help you correct the LLVM IR, I provide the Ghidra decompiled C code for your provided LLVM IR:
```C
{predict_ghidra_c_code}
```
Please compare the original decompiled Ghidra C code with your generated LLVM IR and corresponding Ghidra C code, 
find the differences and correct the LLVM IR.\n
Place the final generated LLVM IR code between ```llvm and ```.
"""

GHIDRA_DECOMPILE_TEMPLATE = """
Please decompile the assembly code to LLVM IR.
```assembly
{asm_code}
```
Here is the C code that is decompiled by Ghidra decompiler for the assembly code:
```C 
{ghidra_c_code}
```
Please reference the Ghidra decompiled C code to decompile the assembly code to LLVM IR.
When reference the Ghidra decompiled C code, please consider the following points:
- The number of parameters of Ghidra is typically correct, but the parameter types may be incorrect, especially for strcut and pointer types.
- The types of Ghidra may be incorrect, especially for strcut and pointer types. please guess the correct types based on the context.
- The overall structure of Ghidra is typically correct, but please check the logic of the code.
- The return type and value of the function of Ghidra is likely to be correct.
When generating the LLVM IR, please consider the following points:
- LLVM IR should follow the Static Single Assignment format, which mean a variable can only be defined once.
- Please think about the LLVM Language Reference Manual to make sure the generated LLVM IR is valid and can be compiled successfully.
- Place the final generated LLVM IR code between ```llvm and ```.
"""

LLM_FIX_PROMPT = """
You have generated the following LLVM IR: ```llvm\n{predict_llvm_ir}```\n
The corresponding assembly code compiled from the generated LLVM IR is: ```assembly\n{predict_assembly}```\n
However your generated LLVM IR is not right.
I have called a LLM to analyze the reason why the generated LLVM IR is not right.
Please correct the LLVM IR based on the following analysis:
{analysis}
Place the final generated LLVM IR code between ```llvm and ```.
"""

# def format_execution_error_prompt_with_angr_debug_trace(first_prompt, predict_llvm_ir, predict_assembly, target_execution_trace, predict_execution_trace)
TEST_ERROR_TEMPLATE_WITH_ANGR_DEBUG_TRACE = """
You previously generated the following LLVM IR:
```llvm
{predict_llvm_ir}
```
I compiled the generated LLVM IR into the following assembly code:
```assembly
{predict_assembly}
```
and executed it.
I then used angr to trace the execution of the assembly code and obtained an instruction-level execution trace.

An example traced instruction is shown below:
```assembly
0x42b2b6: subl %r8d, %edx ; r8d=0x0, rflags=[PF|ZF], edx=0x0->0x0
```
Trace Format Explanation
	•	Each trace entry begins with the instruction address.
	•	This is followed by the instruction mnemonic and operands.
	•	After the ;, register state information is provided.

Register semantics:
	•	If a register is read only, its value at the time of the instruction is shown.
	•	If a register is both read and written, its value is shown in the format:
before_value -> after_value
	•	rflags records the flags after executing the instruction.

Flag meanings:
•	ZF: Zero Flag (result is zero)
•	SF: Sign Flag (result is negative)
•	CF: Carry Flag
•	PF: Parity Flag
•	OF: Overflow Flag
•	AF: Adjust Flag
Below is the execution trace of the ground truth assembly code:
```assembly
{target_execution_trace}
```
Below is the execution trace of your generated assembly code:
```assembly
{predict_execution_trace}
```
Please:
1.	Compare the execution trace of the generated assembly code with the ground truth trace.
2.	Identify all semantic differences (including register values, flags, instruction behavior, missing or extra operations).
3.	Determine the root cause of any divergence.
4.	Correct the LLVM IR so that the compiled assembly matches the ground truth behavior.

Return the corrected LLVM IR only, and place the final version strictly between:
```llvm
and
```
"""
