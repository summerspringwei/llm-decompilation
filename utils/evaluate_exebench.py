
import os
import json
import subprocess
import logging
import pathlib
import shutil
import tqdm
import fire
from typing import Dict, List
from multiprocessing import Pool
from functools import partial
from datasets import load_from_disk

from exebench import Wrapper, diff_io, exebench_dict_to_dict, LLVMAssembler
from utils.extract_code import extract_llmcompiler_code_blocks
from models.assembly_analyzer.extract_asm_function_define import split_elf_functions

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)s - %(message)s ', level=logging.INFO)


def compile_llvm_ir(llvm_ir: str, compile_dir: str, name_hint)->tuple[bool, str]:
    """Compile the llvm_ir to assembly and save the results to the validation directory, return true if success compile
    Args:
        llvm_ir: str, the llvm ir code.
        compile_dir: str, the directory to save the compiled assembly code.
        name_hint: str, the hint for the name of the compiled file.
    
    Returns:
        success: bool, true if the compilation is successful.
        assembly_path: str, the path of the compiled assembly code.
    """
    if not os.path.exists(compile_dir):
        os.makedirs(compile_dir, exist_ok=True)
    llvm_ir_path = os.path.join(compile_dir, f"{name_hint}.ll")
    assembly_path = os.path.join(compile_dir, f"{name_hint}.s")
    object_file_path = os.path.join(compile_dir, f"{name_hint}.o")
    error_path = os.path.join(compile_dir, f"{name_hint}.error")
    success = False
    error_msg = ""
    if isinstance(llvm_ir, list) and len(llvm_ir) == 0:
        logging.error(f"Invalid llvm_ir: {llvm_ir}")
        return success, None, error_msg
    with open(llvm_ir_path, 'w') as f:
        f.write(llvm_ir[0] if isinstance(llvm_ir, list) else llvm_ir)
    
    try:
        # 3. Compile the llvm ir to assembly
        cmd = ["llc", llvm_ir_path, "-o", assembly_path]
        ret = subprocess.run(cmd, capture_output=True)
        cmd = ["llc", llvm_ir_path, "-filetype=obj", "-o", object_file_path]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode == 0:
            success = True
        else:
            # Save the stderr output to the specified file
            with open(error_path, 'w') as f:
                error_msg = ret.stderr.decode()
                f.write(error_msg)
            success = False
    except Exception as e:
        logging.error(e)
        success = False
    return success, assembly_path, error_msg


compile_target_ir = partial(compile_llvm_ir, name_hint="target")
compile_predicted_ir = partial(compile_llvm_ir, name_hint="predict")


def eval_assembly(row: Dict, assembly: str) -> bool:
    """Evaluate the assembly code by running the synthetic test cases.
    
    Args:
        row: Dict, the row of the dataset in exebench.
        assembly: str, the assembly code to be evaluated.
    
    Returns:
        success: bool, true if the evaluation is successful.
    """
    success = True
    synth_wrapper = None
    try:
        c_deps=(row['synth_deps'] + '\n' +
                    row['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
                        'typedef int bool;', '')
        synth_wrapper = Wrapper(
            c_deps=c_deps + '\n',
            func_c_signature=row['func_head_types'].replace('extern', ''),
            func_assembly=assembly,
            cpp_wrapper=row['synth_exe_wrapper'],
            assembler_backend=LLVMAssembler())
        count, total = 0, len(row['synth_io_pairs']['input'])
        for i, o in zip(row['synth_io_pairs']['input'],
                        row['synth_io_pairs']['output']):
            observed_output = synth_wrapper(
                exebench_dict_to_dict(i))  # Run synthetic
            if observed_output is None:
                logging.error('Error: The code could not be compiled')
                success = False
                return success
            # print(observed_output, exebench_dict_to_dict(o))
            count += 1 if diff_io(
                observed_output=observed_output,
                expected_output=exebench_dict_to_dict(o)) else 0
        success = (count == total)
        if not success:
            logging.info(
                f"Error for {row['path']} total cases {total}, success cases {count}"
            )
    except Exception as e:
        logging.error(f"Error for {row['path']} with error_msg: {e}")
        success = False
    finally:
        return success


def has_aarch64_target(llvm_ir: str):
    """Check if the LLVM IR has aarch64 target.
    
    """
    return llvm_ir.find('target triple = "aarch64') != -1


def validate_by_execution(record: Dict, row: Dict, validation_dir:str, target="x86")->Dict:
    # 1. First validate the target llvm IR
    file_path = record['file']
    full_path = os.path.join(validation_dir, file_path, row['fname'])
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    if isinstance(record["output"], list):
        record["output"] = record["output"][0]
    
    target_success, target_assembly_path, error_msg = compile_target_ir(record["output"], full_path)
    target_execution_success = False
    # Validate the target assembly
    if target_success:
        try:
            with open(target_assembly_path, 'r') as f:
                target_execution_success = eval_assembly(row, f.read())
        except Exception as e:
            logging.error(e)
    record["target_compile_success"] = target_success
    record["target_execution_success"] = target_execution_success
    # 2. Validate the predicted assembly
    if isinstance(record["predict"], str):
        record['predict'] = [record['predict'], ]
    if isinstance(record["predict"], list):        
        record["predict_compile_success"] = []
        record["predict_execution_success"] = []
        for predict in record["predict"]:
            # 2.1 Check the llvm ir target
            if target=='x86' and has_aarch64_target(predict):
                record["predict_compile_success"].append(False)
                record["predict_execution_success"].append(False)
                continue
            # 2.2 Compiler the llvm ir to assembly
            predict_success, predict_assembly_path, error_msg = compile_predicted_ir(predict, full_path)
            predict_execution_success = False
            # Validate the predict assembly
            if predict_success:
                try:
                    with open(predict_assembly_path, 'r') as f:
                        predict_execution_success = eval_assembly(row, f.read())
                except Exception as e:
                    logging.error(e)
            record["predict_compile_success"].append(predict_success)
            record["predict_execution_success"].append(predict_execution_success)
        print((record["predict_compile_success"], record["predict_execution_success"], target_success, target_execution_success))
    else:
        logging.error(f"Invalid format of record['predict']: {record['predict']}")
    return record


def wrapper(args):
    if len(args) != 3 or not isinstance(args[0], dict) or not isinstance(args[1], dict):
        logging.error(f"Invalid input: {args}")
        return None
    return validate_by_execution(*args)


def format_path_and_func_def(path: str, func_def: str)->str:
    return str(path)+":"+str(func_def)


def preprocess_records(all_records: list[Dict])->Dict:
    """Preprocess the records to make sure the format is correct.
    Note the record here means the output of the LLM model with instruction and assembly code.

    Parameters:
    all_records: list[Dict], the output of the LLM model with instruction and assembly code.

    Returns:
    path_to_record_mapping: Dict, a mapping from the (file_path:func_def) to the record.
    """
    path_to_record_mapping = {}
    for record in tqdm.tqdm(all_records):
        # Preprocessing the LLM output here:
        if isinstance(record["predict"], str):
            record["predict"] = [record["predict"], ]
        if isinstance(record["predict"], list):
            new_predict_list = []
            for predict in record["predict"]:
                # For llmcompiler, the output is wrapped in code block
                if predict.find("code") >= 0:
                    matched_predict_llvm_ir = extract_llmcompiler_code_blocks(predict)
                    if matched_predict_llvm_ir and len(matched_predict_llvm_ir) > 0:
                        new_predict_list.append(matched_predict_llvm_ir[0])
                    else:
                        # logging.error(f"Cannot find code block in {predict}")
                        logging.error(f"Cannot find code block in {record['file']}")
                        new_predict_list.append(predict)
                else:
                    new_predict_list.append(predict)
                if predict.find("aarch64") >= 0:
                    logging.error(f"Find aarch64 in {record['file']}")
            record["predict"] = new_predict_list
        
        path_to_record_mapping[format_path_and_func_def(record['file'], record["func_head_types"])] = record

    return path_to_record_mapping


def match_record_with_row(path_to_record_mapping: Dict, path_to_row_mapping: Dict):
    # We need also to make sure the function name is the same
    path_to_record_row_mapping = {}
    for path_func_def, record in path_to_record_mapping.items():
        if path_func_def in path_to_row_mapping:
            path_to_record_row_mapping[path_func_def] = (record, path_to_row_mapping[path_func_def])
        else:
            logging.error(f"Cannot find record for {path_func_def}")
    return path_to_record_row_mapping


def extract_wrapper_assembly_functions(row: Dict) -> Dict[str, List[str]]:
    """Extract the wrapper assembly functions from the row.
    Args:
        row: Dict, the row of the dataset in exebench.
    Returns:
        functions: Dict[str, List[str]], the mapping from the function name to the assembly code.
    """
    c_deps=(row['synth_deps'] + '\n' +
                    row['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
                        'typedef int bool;', '')
    assembly = row['asm']['code'][-1]
    assembler = LLVMAssembler()
    wrapper_assembly = assembler.get_wrapper_assembly(c_deps=c_deps + '\n', func_c_signature=row['func_head_types'].replace('extern', ''), func_assembly = assembly,
                 cpp_wrapper=row['synth_exe_wrapper'])
    with open(wrapper_assembly, 'r') as f:
        wrapper_assembly = f.read()
    return split_elf_functions(wrapper_assembly)


def validate_exebench(path_to_json: str = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-step-80-bs-32-beams-1.json", 
                      path_to_dataset: str = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2", 
                      path_to_result: str = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-rl-ppo-step-80-bs-32-beams-1_validate_exebench.json",
                      validation_dir: str = "/home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_validate_exebench"):
    dataset = load_from_disk(
        path_to_dataset
    )
    path_to_row_mapping = {}
    for row in dataset:
        path_to_row_mapping[format_path_and_func_def(row['path'], row["func_head_types"])] = row
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    pathlib.Path(validation_dir).mkdir(parents=True, exist_ok=True)
    all_records = json.load(open(path_to_json, 'r'))
    path_to_record_mapping = preprocess_records(all_records)
    path_to_record_row_mapping = match_record_with_row(path_to_record_mapping, path_to_row_mapping)

    # Run in parallel
    args = [value + (validation_dir,) for _, value in path_to_record_row_mapping.items()]
    with Pool(processes=80) as pool:
        results = pool.map(wrapper, args)
    
    predict_compile_results = [any(r["predict_compile_success"]) if isinstance(r, dict) else False for r in results]
    predict_execution_results = [any(r["predict_execution_success"]) if isinstance(r, dict) else False for r in results]
    target_compile_results = [r["target_compile_success"] if isinstance(r, dict) else False for r in results]
    target_execution_results = [r["target_execution_success"] if isinstance(r, dict) else False for r in results]
    logging.info(f"""Total records: {len(all_records)}, 
                 predict_compile_success:{sum(predict_compile_results)}, 
                 predict_execution_success: {sum(predict_execution_results)},
                 target_compile_success: {sum(target_compile_results)},
                 target_execution_success: {sum(target_execution_results)}""")
    json.dump(results, open(path_to_result, 'w'), indent=4, sort_keys=False, separators=(',', ':'))


if __name__ == "__main__":
    # path_to_json = "bart_exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_json = "deepseek-coder-1.3b-exebench_train_synth_rich_io_filtered_llvm_ir_inc.json"
    # path_to_dataset = "/data/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_ir_assembly_O2"
    # path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json"
    # path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd.json"
    # path_to_json = "exebench_train_synth_rich_io_filtered_llvm_ir_0_llm-compiler-13b-ftd-beams-8.json"
    
    fire.Fire(validate_exebench)
