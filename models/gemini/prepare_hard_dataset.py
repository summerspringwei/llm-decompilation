import os
import json
import subprocess
import tempfile
import fire
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict

"""
Example of function info: {'name': 'foo', 'unused_args': [], 
    'struct_args': False, 'has_globals': True, 
    'called_functions': ['mpt3sas_base_get_iocstate', 'scsi_host_busy', 'wait_event_timeout']}
"""

info_binary = "/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"

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
        f.write(record['llvm_ir']['code'][-1].encode("utf-8"))
        f.flush()
        cmd = [info_binary, f.name]
        cmd_out = subprocess.run(cmd, stdout=subprocess.PIPE)
        llvm_ir_info = cmd_out.stdout.decode("utf-8")
        out = json.loads(llvm_ir_info)
        record['func_info'] = out
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


def main(
    # dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff",
    dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff",
    saved_dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_hard_sample_100",
    bb_threshold = 3
):
    exebench_data = load_from_disk(dataset_path)
    print("exebench_data", len(exebench_data))
    # First map to add functions field
    exebench_data = exebench_data.filter(filter_cannot_parse, num_proc=16, load_from_cache_file=True)
    print("exebench_data after filter_cannot_parse", len(exebench_data))
    dataset_with_func_info = exebench_data.map(map_func_info, num_proc=16, load_from_cache_file=False)
    print("dataset_with_func_info", len(dataset_with_func_info))
    dataset_with_loops = dataset_with_func_info.filter(filter_has_loops, num_proc=16)
    print("dataset_with_loops", len(dataset_with_loops))
    dataset_filter_unused_args = dataset_with_func_info.filter(filter_unused_args,
                                                            num_proc=16)
    print("dataset_filter_unused_args", len(dataset_filter_unused_args))
    # Then filter based on functions length
    dataset_with_called_functions = dataset_filter_unused_args.filter(
        lambda record: len(record['func_info']["functions"][0]["called_functions"]
                        ) > 0,
        num_proc=16)
    print("dataset_with_called_functions", len(dataset_with_called_functions))
    bb_more_than_threshold = dataset_with_called_functions.filter(
        lambda record: record['llvm_ir']['bb_count']["bbcount"] > bb_threshold,
        num_proc=16)
    print("bb_more_than_threshold", len(bb_more_than_threshold))

    dataset_with_struct_args = bb_more_than_threshold.filter(
        filter_with_struct_args,
        num_proc=16)
    print("dataset_with_struct_args", len(dataset_with_struct_args))
    dataset_with_num_of_instructions = dataset_with_struct_args.filter(
        filter_num_of_instructions,
        num_proc=16)
    print("dataset_with_num_of_instructions", len(dataset_with_num_of_instructions))
    # Randomly sample 100 records
    dataset_with_num_of_instructions.save_to_disk(saved_dataset_path)
    dump_llvm_ir(dataset_with_num_of_instructions, "tmp_dir")
    

if __name__ == "__main__":
    fire.Fire(main)
