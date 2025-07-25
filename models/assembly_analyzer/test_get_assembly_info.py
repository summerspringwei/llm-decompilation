
import tempfile
from datasets import load_from_disk
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly, extract_wrapper_assembly_functions
from models.assembly_analyzer.parse_assembly import extract_called_functions
from models.assembly_analyzer.find_call_related import find_call_related
from models.assembly_analyzer.extract_asm_function_define import split_elf_functions
import logging
logging.basicConfig(level=logging.WARNING)
# dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"
dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_hard_sample_100"

dataset = load_from_disk(dataset_dir_path)
for record in dataset:
    # print(record)
    # 1. Extract the called functions from the assembly code
    function_list = extract_called_functions(record['asm']['code'][-1])
    if len(function_list) == 0:
        continue
        # print("Function list:", function_list)
    # predict_execution_success = eval_assembly(record, record['asm']['code'][-1])
    if len(function_list) == 0:
        continue
    # 2. Extract the assembly code of the called functions
    functions_dict = extract_wrapper_assembly_functions(record)
    # print("Functions dict:", functions_dict.keys())
    for func_name in function_list:
        if func_name.find("@PLT") != -1:
            func_name = func_name.split("@PLT")[0]
        if func_name in functions_dict:
            # print(f"Function {func_name} found")
            # print(functions_dict[func_name])
            pass
        else:
            # print(f"Function {func_name} not found in assembly.")
            logging.warning(f"Function {func_name} not found in assembly.")
        # print("==" * 20)
    assert("main" in functions_dict.keys())

    # 3. Get the function name of the decompiled function
    asm_func_name = split_elf_functions(record['asm']['code'][-1]).keys()
    asm_func_name = list(asm_func_name)[0]
    asm_func_name = asm_func_name.split("@PLT")[0] if asm_func_name.find("@PLT") != -1 else asm_func_name
    # 4. Get how the decompiled function is called
    print("called functions:", function_list)
    print("functions in wrapper", functions_dict.keys())
    print("decompiled function:", asm_func_name)

    for func_name, func_impl in functions_dict.items():
        rel = find_call_related(func_impl, asm_func_name, arch="amd64_sysv")
        if len(rel) > 0:
            print(func_name, asm_func_name, rel)
            print("==" * 20)
    # if predict_execution_success:
    #     print("Execution success")
    # else:
    #     print("Execution failed")
