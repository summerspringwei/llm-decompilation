import os
import json
import subprocess
import tempfile
import random
import string
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly

"""
Example of function info: {'name': 'foo', 'unused_args': [], 
    'struct_args': False, 'has_globals': True, 
    'called_functions': ['mpt3sas_base_get_iocstate', 'scsi_host_busy', 'wait_event_timeout']}
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    # Try to find llvm-parser-checker in PATH
    which_cmd = subprocess.run(['which', 'llvm-parser-checker'], 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    if which_cmd.returncode == 0:
        info_binary = which_cmd.stdout.decode('utf-8').strip()
    else:
        info_binary = "/data1/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
except:
    # Keep default value if which command fails
    pass

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

def filter_cannot_parse(record):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
        f.flush()
        cmd = [info_binary, f.name]
        cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
        if cmd_out.returncode != 0:
            return False
    return True


def map_func_info(record):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        try:
            f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
            f.flush()
            cmd = [info_binary, f.name]
            cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
            llvm_ir_info = cmd_out.stdout.decode("utf-8")
            out = json.loads(llvm_ir_info)
            # Set default values for any potentially missing fields
            
            if out is None or 'functions' not in out:
                print(record['llvm_ir']['code'][-1])
                print("error_ir_info:", llvm_ir_info)
                out = {
                    'functions': [{
                        'called_functions': [],
                        'has_defined_structs': False,
                        'has_globals': False,
                        'name': '',
                        'num_loops': 0,
                        'struct_args': False,
                        'unused_args': []
                    }]
                }
        except Exception as e:
            print(e)
            out = {
                'functions': [{
                    'called_functions': [],
                    'has_defined_structs': False,
                    'has_globals': False,
                    'name': '',
                    'num_loops': 0,
                    'struct_args': False,
                    'unused_args': []
                }]
            }
        
        finally:
            record['func_info'] = out
            tokenized_question = tokenizer(record['asm']['code'][-1])
            record['token_length'] = len(tokenized_question["input_ids"])
    
        return record


def filter_unused_args(record):
    for func_dict in record['func_info']['functions']:
        has_unused_args = True if len(func_dict['unused_args']) > 0 else False
        if has_unused_args:
            return False
    return True


def filter_with_struct_args(record):
    for func_dict in record['func_info']['functions']:
        if func_dict['has_defined_structs']:
            return True
    return False


def dump_llvm_ir(exebench_data, saved_dataset_path):
    for idx, record in enumerate(exebench_data):
        with open(os.path.join(saved_dataset_path, f"{idx}.ll"), "w") as f:
            f.write(f";cpath: {record['path']}\n" + record['llvm_ir']['code'][-1])


def filter_num_of_instructions(record, threshold=50):
    filter_num_of_instructions = sum(record['llvm_ir']['bb_count']['bb_list_size'])
    return True if filter_num_of_instructions > threshold else False


def filter_has_loops(record):
    for func_dict in record['func_info']['functions']:
        if func_dict['num_loops'] > 0:
            return True
    return False


def filter_has_function_call(record):
    for func_dict in record['func_info']['functions']:
        if len(func_dict['called_functions']) > 0:
            return True
    return False


def random_sample_dir(base_dir="/tmp/sample_dir"):
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return os.path.join(base_dir, rand_str)


def filter_record_execution_success(record):
    sample_dir = random_sample_dir()
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
    target_compile_success, target_assembly_path = compile_llvm_ir(record["llvm_ir"]["code"][-1], sample_dir, name_hint="target")
    if target_compile_success:
        with open(target_assembly_path, 'r') as f:
            target_execution_success = eval_assembly(record, f.read())
    subprocess.run(["rm", "-rf", sample_dir])
    return target_execution_success
