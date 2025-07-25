import os
import re
import json
import pickle
import logging
import subprocess

from openai import OpenAI
# from google import genai
from functools import partial
from datasets import load_from_disk, Dataset
from multiprocessing import Pool
from utils.evaluate_exebench import compile_llvm_ir
from utils.preprocessing_assembly import preprocessing_assembly
from utils.openai_helper import extract_llvm_code_from_response, format_decompile_prompt, format_compile_error_prompt, format_execution_error_prompt
import tempfile

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                timeout=1800)
model_name = 'ep-20250317013717-m9ksl'
USE_CACHE = True

def huoshan_deepseek_r1(client, prompt: str):
    response = client.chat.completions.create(model=model_name,
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }],
                                              stream=False)
    return response


def run_command(command, cwd):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        logging.warning(
            f"Command failed: {' '.join(command)}\nError: {result.stderr}")
    return result.returncode


def parse_test_results(output: str) -> dict:
    result_dict = {}
    pattern = r"# (\w+):\s+(\d+)"
    matches = re.findall(pattern, output)
    for key, value in matches:
        result_dict[key] = int(value)
    return result_dict



def one_pass(client, prompt: str, output_dir: str, module_name: str, func_name: str, count: int, cwd: str) -> str:
    # 0. Get response from the model
    response_file_path = os.path.join(
        output_dir, f"response_{func_name}_retry_{count}.pkl")
    predict_compile_success = True
    predict_execution_success = True
    if not USE_CACHE:
        response = huoshan_deepseek_r1(client, prompt)
        pickle.dump(response, open(response_file_path, "wb"))
    else:
        response = pickle.load(open(response_file_path, "rb"))
    predict_llvm_ir = extract_llvm_code_from_response(response)
    # TODO: If the response is empty, we need to set a error message to retry
    if len(predict_llvm_ir) == 0:
        logger.warning(
            f"Empty prediction for {func_name} on retry {count}")
        return "Empty prediction"
    # 1. Try to compile the decompiled LLVM IR with `llc`
    name_hint = f"src/{module_name}_predict_{func_name}"
    predict_compile_success, assembly_path = compile_llvm_ir(
        predict_llvm_ir, output_dir, name_hint)
    predict_assembly = ""
    if predict_compile_success:
        with open(assembly_path, 'r') as f:
            predict_assembly = f.read()
    error_msg = ""
    if not predict_compile_success:
        assert (os.path.exists(
            os.path.join(output_dir, f"{name_hint}.error")))
        with open(os.path.join(output_dir, f"{name_hint}.error"),
                    'r') as f:
            error_msg = f.read()
        logger.warning(
            f"Compilation failed for {func_name} on retry {count}: {error_msg}"
        )
        return {
            "compile_success": False,
            "execution_success": False,
            "error_message": error_msg,
            "predict_llvm_ir": predict_llvm_ir,
            "predict_assembly": predict_assembly
        }
    # 3. try to link the decompiled ir with the original one
    # link and compile
    cmd = [
        "llvm-link", f"src/{module_name}_no_{func_name}.ll",
        f"src/{module_name}_predict_{func_name}.ll", "-o", f"src/{module_name}_predict.ll"
    ]
    results = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                cwd=cwd,
                                # executable="/bin/bash",Figure out the effect of this
                                )
    if results.returncode != 0:
        logger.warning(f"Error linking files: {results.stderr}")
        return {
            "compile_success": False,
            "execution_success": False,
            "error_message": results.stderr,
            "predict_llvm_ir": predict_llvm_ir,
            "predict_assembly": predict_assembly
        }
    # 4. Compile the linked LLVM IR to object file
    target_object_file = f"src/{module_name}_predict.o"
    cmd = [
        "clang", "-c", f"src/{module_name}_predict.ll", "-o", target_object_file
    ]
    results = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                cwd=cwd)
    if results.returncode != 0:
        logger.warning(f"Error compiling files: {results.stderr}")
        return {
            "compile_success": False,
            "execution_success": False,
            "error_message": results.stderr,
            "predict_llvm_ir": predict_llvm_ir,
            "predict_assembly": predict_assembly
        }
    predict_compile_success = True
    # 5. Link the object file to the executable binary
    target_binary_path = os.path.join(cwd, f"src/{module_name}")
    if os.path.exists(target_binary_path):
        os.remove(target_binary_path)
    run_command([
        "clang", "-Wno-format-extra-args",
        "-Wno-implicit-const-int-float-conversion",
        "-Wno-tautological-constant-out-of-range-compare", "-g", "-O2",
        "-Wl,--as-needed", "-o", target_binary_path, target_object_file,
        "src/iopoll.o", "src/libver.a", "lib/libcoreutils.a",
        "lib/libcoreutils.a", "-ldl"
    ], cwd)
    # 6. Run the test
    cmd = [
        "make", "check",
        f"TESTS=\"$(make listtests | tr ' ' '\\n' | grep '^tests/{module_name}')\""
    ]
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as temp_script:
        temp_script.write("#!/bin/bash\n")
        temp_script.write("set -e\n")
        temp_script.write(f"cd {cwd}\n")
        temp_script.write(" ".join(cmd) + "\n")
    temp_script_path = temp_script.name
    os.chmod(temp_script_path, 0o755)
    cmd = ["bash", temp_script_path]
    results = subprocess.run(cmd, capture_output=True, text=True)
    # os.unlink(temp_script_path)
    logger.info(f"Test result: {results.stdout}")
    # 7. Check the test result
    test_result = parse_test_results(results.stdout)
    predict_execution_success = (test_result[
            "TOTAL"] == test_result["PASS"] + test_result["SKIP"]
            and test_result["FAIL"] == 0 and test_result["TOTAL"] > 0)
    if predict_execution_success:
        logger.info(
            f"Decompilation successful for {func_name} on retry {count}"
        )
        return {
            "compile_success": True,
            "execution_success": True,
            "error_message": "",
            "predict_llvm_ir": predict_llvm_ir,
            "predict_assembly": predict_assembly
        }
    else:
        return {
            "compile_success": predict_compile_success,
            "execution_success": predict_execution_success,
            "error_message": "",
            "predict_llvm_ir": predict_llvm_ir,
            "predict_assembly": predict_assembly
        }


def decompilation_loop(client,
                       module_name: str,
                       func_name: str,
                       output_dir: str,
                       num_retry: int = 10,
                       remove_comments: bool = True,
                       cwd="/home/xiachunwei/Projects/coreutils"):
    with open(os.path.join(output_dir, f"src/{module_name}_target_{func_name}.s"),
              'r') as f:
        asm_code = f.read()
    asm_code = preprocessing_assembly(asm_code,
                                      remove_comments=remove_comments)

    prompt = format_decompile_prompt(asm_code)
    count = 0
    predict_compile_success, predict_execution_success = False, False
    while count < num_retry and (not predict_compile_success
                                 or not predict_execution_success):
        count += 1
        logger.info(f"Retrying {count} times for {func_name}")
        try:
            info_dict = one_pass(client, prompt, output_dir, module_name, func_name, count, cwd)
            print(info_dict['error_message'])
            predict_compile_success, predict_execution_success = info_dict[
                "compile_success"], info_dict["execution_success"]
            if not predict_compile_success:
                prompt = format_compile_error_prompt(
                    asm_code, info_dict["predict_llvm_ir"], info_dict["error_message"])
            elif not predict_execution_success:
                prompt = format_execution_error_prompt(
                    asm_code, info_dict["predict_llvm_ir"], info_dict["predict_assembly"])
            else:
                logger.info(
                    f"Decompilation successful for {func_name} on retry {count}"
                )
                break
        except Exception as e:
            logging.warning(f"Error during decompilation: {e}")


def auto_test_one_function(module_name: str,
                           func_name: str,
                           cwd: str ="/home/xiachunwei/Projects/coreutils"):
    # Step 1: Extract one function from LLVM IR file
    if run_command([
            "llvm-extract", f"--func={func_name}", "-S", f"src/{module_name}.ll", "-o",
            f"src/{module_name}_target_{func_name}.ll"
    ], cwd) != 0:
        return

    # Step 2: Delete one function from module
    if run_command([
            "llvm-extract", "-delete", f"--func={func_name}", "-S",
            f"src/{module_name}.ll", "-o", f"src/{module_name}_no_{func_name}.ll"
    ], cwd) != 0:
        return

    # Step 3: Compile function LLVM IR to assembly
    if run_command([
            "llc", f"src/{module_name}_target_{func_name}.ll", "-o",
            f"src/{module_name}_target_{func_name}.s"
    ], cwd) != 0:
        return

    # Step 4: Use the LLM to decompile assembly to IR
    # Placeholder for LLM decompilation logic (not specified in the prompt)
    decompilation_loop(client,
                       module_name,
                       func_name,
                       output_dir=cwd,
                       num_retry=3,
                       remove_comments=True,
                       cwd=cwd)


INFO_BINARY = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
def get_all_function_names(llvm_ir_file):
    cmd = [INFO_BINARY, llvm_ir_file]
    cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
    if cmd_out.returncode != 0:
        logger.warning(
            f"Error Counting bb for: {llvm_ir_file} output: {cmd_out.stdout}, error: {cmd_out.stderr}")
        return
    llvm_ir_info = cmd_out.stdout.decode("utf-8")
    out = json.loads(llvm_ir_info)
    func_name_list = []
    for func_dict in out["functions"]:
        for func_name, func_info in func_dict.items():
            func_name_list.append(func_name)
    print(func_name_list)
    return func_name_list


COREUTILS_DIR = "/home/xiachunwei/Projects/coreutils"

def test_decompilation_one_module(module_name: str):
    func_name_list = get_all_function_names(
        os.path.join(COREUTILS_DIR, f"src/{module_name}.ll"))
    for func_name in func_name_list:
        if func_name == "main":
            # Skip the main function and pretty_name
            continue
        if func_name in ["recheck", "any_symlinks", "check_fspec"]:
            auto_test_one_function(module_name, func_name, cwd=COREUTILS_DIR)

def test():
    # auto_test_one_function("pretty_name")
    # get_all_function_names(
    #     "/home/xiachunwei/Projects/coreutils/src/tail.ll")
    test_decompilation_one_module("tail")


if __name__ == "__main__":
    test()
