from openai import OpenAI
import pickle
import re, os
from multiprocessing import Pool
from datasets import load_from_disk, concatenate_datasets
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.exebench_dataset_processing import filter_has_loops, map_func_info, filter_record_execution_success
from typing import List
# Set OpenAI's API key and API base to use vLLM's API server.
from models.gemini.gemini_decompilation import evaluate_response

def merge_datasets(dataset_dir):
    dataset_list = []
    for idx in range(7):
        dataset_path = os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")
        dataset = load_from_disk(dataset_path)
        dataset_list.append(dataset)

    dataset = concatenate_datasets(dataset_list)
    return dataset


def filter_length(record):
    return record['token_length'] < 4096


def get_idx_list(response_output_dir):
    results = pickle.load(open(os.path.join(response_output_dir, "results.pkl"), "rb"))
    tmp_onebb_idx_list = []
    for idx, (response, validation_results) in enumerate(results):
        if "target_execution_success" not in validation_results.keys() or not validation_results["target_execution_success"]:
            tmp_onebb_idx_list.append(idx)
    return tmp_onebb_idx_list


def main(dataset_dir = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/",
         output_dir = "/home/xiachunwei/Datasets/filtered_exebench",
         num_proc = 40):

    dataset = merge_datasets(dataset_dir)

    dataset = dataset.map(
        map_func_info,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    dataset = dataset.filter(
        filter_length,
        num_proc=num_proc,
        # load_from_cache_file=False,
    )

    dataset_loops = dataset.filter(
        filter_has_loops,
        num_proc=num_proc,
        # load_from_cache_file=False,
    )
    dataset_loops = dataset_loops.select(range(100))
    dataset_loops = dataset_loops.filter(
        filter_record_execution_success,
        num_proc=num_proc,
    )
    print("len(dataset_loops)", len(dataset_loops))
    # Create dataset without loops
    dataset_without_loops = dataset.filter(
        lambda x: not filter_has_loops(x),
        num_proc=num_proc,
    )
    print("len(dataset_without_loops)", len(dataset_without_loops))
    dataset_without_loops = dataset_without_loops.select(range(100))
    dataset_without_loops = dataset_without_loops.filter(
        filter_record_execution_success,
        num_proc=num_proc,
    )
    dataset_with_only_one_bb = dataset.filter(
        lambda x: x['llvm_ir']['bb_count']['bbcount'] == 1,
        num_proc=num_proc,
    )
    print("len(dataset_with_only_one_bb)", len(dataset_with_only_one_bb))
    dataset_with_only_one_bb = dataset_with_only_one_bb.select(range(100))
    dataset_with_only_one_bb = dataset_with_only_one_bb.filter(
        filter_record_execution_success,
        num_proc=num_proc,
    )

    print("finished filtering")
    
    dataset_paires = [
        ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops", "validation/qwen3-32b/sample_without_loops_qwen3-32b-n8-assembly-with-comments"),
        ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops", "validation/qwen3-32b/sample_loops_qwen3-32b-n8-assembly-with-comments"),
        ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb", "validation/qwen3-32b/sample_only_one_bb_qwen3-32b-n8-assembly-with-comments"),
    ]
    without_loops_dataset = load_from_disk(dataset_paires[0][0])
    without_loops_idx_list = get_idx_list(dataset_paires[0][1])
    loops_dataset = load_from_disk(dataset_paires[1][0])
    loops_idx_list = get_idx_list(dataset_paires[1][1])
    one_bb_dataset = load_from_disk(dataset_paires[2][0])
    one_bb_idx_list = get_idx_list(dataset_paires[2][1])
    
    current_without_loops_path = [record['path'] for record in dataset_without_loops]
    for idx in without_loops_idx_list:
        found = False
        for record in dataset_without_loops:
            if record['path'] not in current_without_loops_path:
                # To replace the record at idx in a HuggingFace Dataset, use the .select and .add_item methods are not available.
                # Instead, convert to a list, replace, and reconstruct the Dataset:
                data = list(without_loops_dataset)
                data[idx] = record
                from datasets import Dataset
                without_loops_dataset = Dataset.from_list(data)
                found = True
                break
        if found:
            break
    print("finished without loops", without_loops_idx_list)
    current_loops_path = [record['path'] for record in loops_dataset]
    for idx in loops_idx_list:
        found = False
        for record in dataset_loops:
            if record['path'] not in current_loops_path:
                data = list(loops_dataset)
                data[idx] = record
                from datasets import Dataset
                loops_dataset = Dataset.from_list(data)
                found = True
                break
        if found:
            break
    print("finished loops", loops_idx_list)
    current_one_bb_path = [record['path'] for record in one_bb_dataset]
    for idx in one_bb_idx_list:
        found = False
        for record in dataset_with_only_one_bb:
            if record['path'] not in current_one_bb_path:
                data = list(one_bb_dataset)
                data[idx] = record
                from datasets import Dataset
                one_bb_dataset = Dataset.from_list(data)
                found = True
                break
        if found:
            break
    print("finished one bb", one_bb_idx_list)
    without_loops_dataset.save_to_disk(dataset_paires[0][0]+"_164")
    loops_dataset.save_to_disk(dataset_paires[1][0]+"_164")
    one_bb_dataset.save_to_disk(dataset_paires[2][0]+"_164")


if __name__ == "__main__":
    main()
