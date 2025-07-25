import os
import tqdm
import numpy as np
import fire
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets
from utils.exebench_dataset_processing import filter_has_loops, map_func_info, filter_record_execution_success, filter_has_function_call

# Enable parallelism for the tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_matching_distribution_worker(record, dataset_without_loops):
    """This is used to match loops and non-loops, they must have the same (or similar) number of BBs
    and the same (or similar) token length
    
    Args:
        record: the record to match
        dataset_without_loops: the dataset to match against
        
    Returns:
        the index of the matched record
    """
    token_length = record['token_length']
    bb_count = record['llvm_ir']['bb_count']['bbcount']
    found = False
    bb_threshold = 1
    token_threshold = 0.1
    found_idx = 0
    while not found:
        for idx, to_select_record in enumerate(dataset_without_loops):
            if abs(to_select_record['llvm_ir']['bb_count']['bbcount'] - bb_count) < bb_threshold and \
                abs(to_select_record['token_length'] - token_length) < token_threshold * token_length:
                found = True
                found_idx = idx
                break
        if not found:
            bb_threshold += 1
            token_threshold += 0.1
    return found_idx


def sample_matching_distribution(dataset_with_loops, dataset_without_loops, n_samples=164, num_proc=40):
    # First, randomly sample from dataset with loops
    indices_with_loops = np.random.choice(len(dataset_with_loops), n_samples, replace=False)
    sampled_with_loops = dataset_with_loops.select(indices_with_loops)
    args = [(record, dataset_without_loops) for record in sampled_with_loops]
    
    with Pool(processes=num_proc) as pool:
        results = pool.starmap(sample_matching_distribution_worker, args)
    sampled_without_loops = dataset_without_loops.select(results)
    
    return sampled_with_loops, sampled_without_loops


def match_loops_with_only_one_bb(record, to_be_select_dataset):
    """This is used to match loops and non-loops, they must have the same (or similar) number of BBs
    and the same (or similar) token length
    
    Args:
        dataset_with_loops: the dataset with loops
        dataset_with_only_one_bb: the dataset with only one bb
        
    Returns:
        the index of the matched record
    """
    token_length = record['token_length']
    found = False
    token_threshold = 0.01
    found_idx = 0
    while not found:
        for idx, to_select_record in enumerate(to_be_select_dataset):
            if abs(to_select_record['token_length'] - token_length) < token_threshold * token_length:
                found = True
                found_idx = idx
                break
        if not found:
            token_threshold += 0.01
    return found_idx


def match_loops_with_only_one_bb_batch(dataset_with_loops, to_be_select_dataset, num_proc=40):
    args = [(record, to_be_select_dataset) for record in dataset_with_loops]
    with Pool(processes=num_proc) as pool:
        results = pool.starmap(match_loops_with_only_one_bb, args)
    return to_be_select_dataset.select(results)


def plot_token_length_distribution(reference_dataset, to_be_select_dataset, save_path):
    token_lengths_reference = [record['token_length'] for record in reference_dataset]
    token_lengths_to_be_select = [record['token_length'] for record in to_be_select_dataset]
    plt.hist(token_lengths_reference, bins=10, alpha=0.5, label='Reference', density=True)
    plt.hist(token_lengths_to_be_select, bins=10, alpha=0.5, label='To Be Select', density=True)
    plt.title('Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_bb_count_distribution(reference_dataset, to_be_select_dataset, save_path):
    bb_counts_reference = [record['llvm_ir']['bb_count']['bbcount'] for record in reference_dataset]
    bb_counts_to_be_select = [record['llvm_ir']['bb_count']['bbcount'] for record in to_be_select_dataset]
    plt.hist(bb_counts_reference, bins=30, alpha=0.5, label='Reference', density=True)
    plt.hist(bb_counts_to_be_select, bins=30, alpha=0.5, label='To Be Select', density=True)
    plt.title('BB Count Distribution')
    plt.xlabel('BB Count')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def filter_length(record):
    return record['token_length'] < 4096


def merge_datasets(dataset_dir):
    dataset_list = []
    for idx in range(7):
        dataset_path = os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")
        dataset = load_from_disk(dataset_path)
        dataset_list.append(dataset)

    dataset = concatenate_datasets(dataset_list)
    return dataset


def main(dataset_dir = "/home/xiachunwei/Datasets/filtered_exebench/filtered_exebench/",
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

    # Create dataset without loops
    dataset_without_loops = dataset.filter(
        lambda x: not filter_has_loops(x),
        num_proc=num_proc,
    )

    dataset_with_only_one_bb = dataset.filter(
        lambda record: record['llvm_ir']['bb_count']['bbcount'] == 1,
        num_proc=num_proc,
    )

    dataset_with_only_one_bb_function_call = dataset.filter(
        filter_has_function_call,
        num_proc=num_proc,
    )
    dataset_with_only_one_bb_function_call = dataset_with_only_one_bb_function_call.filter(
        filter_record_execution_success,
        num_proc=num_proc,
    )

    dataset_with_only_one_bb_without_function_call = dataset.filter(
        lambda record: not filter_has_function_call(record),
        num_proc=num_proc,
    )   
    dataset_with_only_one_bb_without_function_call = dataset_with_only_one_bb_without_function_call.filter(
        filter_record_execution_success,
        num_proc=num_proc,
    )
    print(f"dataset_with_only_one_bb_function_call length: {len(dataset_with_only_one_bb_function_call)}")
    print(f"dataset_with_only_one_bb_without_function_call length: {len(dataset_with_only_one_bb_without_function_call)}")
    print("finished filtering")

    # # Sample datasets with matching distributions
    sampled_with_loops, sampled_without_loops = sample_matching_distribution(dataset_loops, dataset_without_loops)
    # # # Plot the distributions to verify matching
    # plot_token_length_distribution(dataset_with_only_one_bb, dataset_without_loops, "figures/loops_non_loops_token_length_distribution.png")
    # plot_bb_count_distribution(dataset_with_only_one_bb, dataset_without_loops, "figures/loops_non_loops_bb_count_distribution.png")
    # # Save the sampled datasets
    # sampled_with_loops.save_to_disk(os.path.join(output_dir, 'sampled_dataset_with_loops'))
    # sampled_without_loops.save_to_disk(os.path.join(output_dir, 'sampled_dataset_without_loops'))
    # matched_loops = match_loops_with_only_one_bb_batch(sampled_with_loops, dataset_with_only_one_bb, num_proc=num_proc)
    # matched_loops.save_to_disk(os.path.join(output_dir, "sampled_dataset_with_loops_and_only_one_bb"))
    # print(f"matched_loops length: {len(matched_loops)}")
    # print_dataset_info(matched_loops)
    matched_on_bb_function_call = match_loops_with_only_one_bb_batch(sampled_with_loops, dataset_with_only_one_bb_function_call)
    matched_on_bb_function_call.save_to_disk(os.path.join(output_dir, "matched_on_bb_function_call"))
    plot_token_length_distribution(sampled_with_loops, matched_on_bb_function_call, "figures/loops_on_bb_function_call_token_length_distribution.png")
    matched_on_bb_without_function_call = match_loops_with_only_one_bb_batch(sampled_with_loops, dataset_with_only_one_bb_without_function_call)
    matched_on_bb_without_function_call.save_to_disk(os.path.join(output_dir, "matched_on_bb_without_function_call"))
    plot_token_length_distribution(sampled_with_loops, matched_on_bb_without_function_call, "figures/loops_on_bb_without_function_call_token_length_distribution.png")



def print_dataset_info(dataset):
    token_lengths = sorted([record['token_length'] for record in dataset])
    bb_counts = sorted([record['llvm_ir']['bb_count']['bbcount'] for record in dataset])
    print(f"Token length : {token_lengths}")
    print(f"BB count : {bb_counts}")


def print_filtered_dataset_info():
    sample_with_loops = load_from_disk("sampled_dataset_with_loops")
    sample_without_loops = load_from_disk("sampled_dataset_with_loops_and_only_one_bb")
    print(f"sample_with_loops length: {len(sample_with_loops)}")
    print(f"sample_without_loops length: {len(sample_without_loops)}")
    print_dataset_info(sample_with_loops)
    print_dataset_info(sample_without_loops)
    plot_token_length_distribution(sample_with_loops, sample_without_loops, "matched_loops_token_length_distribution.png")


if __name__ == "__main__":
    fire.Fire(main)
    # dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/"
    # dataset_list = ["sampled_dataset_with_loops", "sampled_dataset_without_loops", "sampled_dataset_with_loops_and_only_one_bb"]
    
    # print_filtered_dataset_info()
