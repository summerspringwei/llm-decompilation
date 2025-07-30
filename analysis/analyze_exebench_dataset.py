from functools import cache
import os
import tempfile
import subprocess
import fire
from matplotlib import pyplot as plt
from datasets import load_dataset
from utils.exebench_dataset_processing import map_func_info
from analysis.filter_exebench_basedon_bb import filter_by_llvm_diff
import pickle
import numpy as np
import json
import re
LLVM_PARSER_CHECKER = "/data1/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker"
OPT_LEVEL = "-O2"

def remove_static_from_function(func_def):
    """
    Check if a function is declared with static, inline, __inline, or __inline__ and remove them if present.
    Args:
        func_def: String containing the function definition
    Returns:
        Modified function definition with static, inline, __inline, and __inline__ removed if they were present
    """
    # Pattern to match function declarations with static and/or inline variants
    # This handles various cases like:
    # static int func() { ... }
    # static inline int func() { ... }
    # inline static int func() { ... }
    # inline int func() { ... }
    # __inline int func() { ... }
    # __inline__ int func() { ... }
    # static __inline int func() { ... }
    # static __inline__ int func() { ... }
    # __inline static int func() { ... }
    # __inline__ static int func() { ... }
    # static int* func() { ... }
    # static struct type func() { ... }
    pattern = r'^\s*(?:static\s+)?(?:inline\s+|__inline\s+|__inline__\s+)?(?:static\s+)?(?:[\w\s*<>]+)\s+\w+\s*\('
    
    # Check if the function starts with static or inline variants
    if re.match(pattern, func_def, re.MULTILINE):
        # Remove static, inline, __inline, and __inline__ keywords and any extra whitespace
        modified_def = re.sub(r'^\s*(?:static\s+|inline\s+|__inline\s+|__inline__\s+)+', '', func_def, flags=re.MULTILINE)
        return modified_def
    return func_def

def map_func_to_llvm_ir(record):
    cpp_code = record["synth_deps"] + "\n" + remove_static_from_function(record["func_def"])
    record["llvm_ir"] = {}
    with tempfile.NamedTemporaryFile(delete=True, suffix=".c") as fcpp:
        cpp_code = remove_static_from_function(cpp_code)
        fcpp.write(cpp_code.encode('utf-8'))
        fcpp.flush()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".ll") as fll:
            cmd = f"clang -std=c11 -S -Wno-int-conversion -emit-llvm {OPT_LEVEL} {fcpp.name} -o {fll.name}"
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                print(f"Error: {cmd}")
                os.system(f"touch {fll.name}")
                record["llvm_ir"]['code'] = [""]
            else:
                with open(fll.name, "r") as f:
                    record["llvm_ir"]['code'] = [f.read()]
                cmd = [LLVM_PARSER_CHECKER, fll.name]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                func_info = json.loads(ret.stdout)
                if len(func_info['functions']) == 0:
                    print(cpp_code)
        
        # Generate LLVM IR with O0 to help extract function info
        with tempfile.NamedTemporaryFile(delete=True, suffix=".ll") as fll:
            cmd = f"clang -std=c11 -S -Wno-int-conversion -emit-llvm -O0 {fcpp.name} -o {fll.name}"
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                print(f"Error: {cmd}")
                os.system(f"touch {fll.name}")
                record["llvm_ir_X86_O0"] = ""
            else:
                with open(fll.name, "r") as f:
                    record["llvm_ir_X86_O0"] = f.read()
    return record


def filter_func_to_llvm_ir(record):
    cpp_code = record["synth_deps"] + "\n" + remove_static_from_function(record["func_def"])
    record["llvm_ir"] = {}
    with tempfile.NamedTemporaryFile(delete=True, suffix=".c") as fcpp:
        cpp_code = remove_static_from_function(cpp_code)
        fcpp.write(cpp_code.encode('utf-8'))
        fcpp.flush()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".ll") as fll:
            cmd = f"clang -std=c11 -S -Wno-int-conversion -emit-llvm {OPT_LEVEL} {fcpp.name} -o {fll.name}"
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                print(f"Error: {cmd}")
                os.system(f"touch {fll.name}")
                record["llvm_ir"]['code'] = [""]
            else:
                with open(fll.name, "r") as f:
                    record["llvm_ir"]['code'] = [f.read()]
                cmd = [LLVM_PARSER_CHECKER, fll.name]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f"Error: {cmd}")
                    return False
                try:
                    func_info = json.loads(ret.stdout)
                    if len(func_info['functions']) == 0:
                        print(cpp_code)
                        return False
                except Exception as e:
                    print(e)
                    return False
    return True


def filter_invalid_records(record):
    if len(record["llvm_ir"]['code'][-1]) == 0:
        return False
    return True


def map_ir_to_asm(record):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".ll") as fll:
        fll.write(record["llvm_ir"]['code'][-1].encode('utf-8'))
        fll.flush()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".s") as fassembly:
            cmd = f"llc {fll.name} -o {fassembly.name}"
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                record["asm"]['code'] = [""]
                return record
            else:
                with open(fassembly.name, "r") as f:
                    record["asm"]['code'] = [f.read()]
    return record

def filter_invalid_asm(record):
    if len(record["asm"]['code'][-1]) == 0:
        return False
    return True


def prepare_dataset(dataset, num_proc=80):
    dataset = dataset.filter(filter_func_to_llvm_ir, num_proc=num_proc)
    print(len(dataset))
    dataset = dataset.map(map_func_to_llvm_ir, num_proc=num_proc, load_from_cache_file=True)
    print(len(dataset))
    dataset = dataset.filter(filter_invalid_records, num_proc=num_proc)
    print(len(dataset))
    dataset = dataset.map(map_ir_to_asm, num_proc=num_proc)
    print(len(dataset))
    dataset = dataset.filter(filter_invalid_asm, num_proc=num_proc)
    print(len(dataset))
    dataset = dataset.map(map_func_info, num_proc=1, load_from_cache_file=True)
    print(len(dataset))
    return dataset


def filter_defined_struct(record):
    if record['func_info'] and 'functions' in record['func_info']:
        for func in record['func_info']['functions']:
            if func['has_defined_structs']:
                return True
    return False


def filter_not_all_struct_field_access(record):
    if record['func_info'] and 'functions' in record['func_info']:
        for func in record['func_info']['functions']:
            if func['has_defined_structs']:
                if not func['all_struct_fields_accessed']:
                    return True
    return False


def count_records_with_min_bbs(dataset, n):
    count = 0
    for record in dataset:
        if record['func_info'] and 'functions' in record['func_info']:
            for func in record['func_info']['functions']:
                if len(func['bbcount']) >= n:
                    count += 1
                    break  # Count each record only once
    print(f"Number of records with {n}+ basic blocks: {count} ({(count/len(dataset))*100:.2f}%)")
    return count


def filter_func_with_min_bbs(record, max_bbs=20):
    if record['func_info'] and 'functions' in record['func_info']:
        for func in record['func_info']['functions']:
            if len(func['bbcount']) <= max_bbs:
                return True
    return False


def plot_bb_distribution(dataset, title, save_path):
    bb_counts = []
    for record in dataset:
        if record['func_info'] and 'functions' in record['func_info']:
            for func in record['func_info']['functions']:
                bb_counts.append(len(func['bbcount']))
    
    plt.figure(figsize=(10, 6))
    plt.hist(bb_counts, bins=30, alpha=0.7)
    plt.title(f'Basic Block Count Distribution - {title}')
    plt.xlabel('Number of Basic Blocks per Function')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(save_path)
    plt.close()



def plot_combined_bb_distribution(valid_dataset, test_dataset, save_path):
    valid_bb_counts = []
    test_bb_counts = []

    # Collect counts for valid set
    for record in valid_dataset:
        if record['func_info'] and 'functions' in record['func_info']:
            for func in record['func_info']['functions']:
                valid_bb_counts.append(len(func['bbcount']))

    # Collect counts for test set
    for record in test_dataset:
        if record['func_info'] and 'functions' in record['func_info']:
            for func in record['func_info']['functions']:
                test_bb_counts.append(len(func['bbcount']))

    # Get the bins for both datasets
    all_counts = valid_bb_counts + test_bb_counts
    if len(all_counts) == 0:
        print("No data to plot.")
        return
        
    min_count = min(all_counts)
    max_count = max(all_counts)
    bins = np.arange(min_count, max_count + 2) - 0.5  # bin edges for integer bins

    valid_hist, _ = np.histogram(valid_bb_counts, bins=bins)
    test_hist, _ = np.histogram(test_bb_counts, bins=bins)
    x = np.arange(min_count, max_count + 1)

    width = 0.4  # width of the bars

    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, valid_hist, width=width, label='Validation Set', color='blue', align='center')
    plt.bar(x + width/2, test_hist, width=width, label='Test Set', color='red', align='center')
    plt.title('Basic Block Count Distribution - Side by Side')
    plt.xlabel('Number of Basic Blocks per Function')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(save_path)
    plt.close()


def read_bb_from_project(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        function_infos = pickle.load(f)
    
    # Extract bbcount lengths (number of basic blocks per function) and filter for <= 21
    # bb_counts = [len(func['bbcount']) for func in function_infos if 'bbcount' in func and len(func['bbcount']) <= 21]
    bb_counts = [len(func['bbcount']) for func in function_infos if 'bbcount' in func]
    return bb_counts


def plot_bb_for_a_series_of_bbcounts(bb_count_list, bb_count_names):
    # Get the bins for all datasets
    for bb_count, name in zip(bb_count_list, bb_count_names):
        print(name, len(bb_count))
    
    # Filter counts to only include those <= 20
    filtered_bb_count_list = []
    for counts in bb_count_list:
        filtered_counts = [count for count in counts if count <= 20]
        filtered_bb_count_list.append(filtered_counts)
    
    all_counts = []
    for counts in filtered_bb_count_list:
        all_counts.extend(counts)
    
    if len(all_counts) == 0:
        print("No data to plot.")
        return
        
    min_count = min(all_counts)
    max_count = 20  # Force max count to be 20
    bins = np.arange(min_count, max_count + 2) - 0.5  # bin edges for integer bins
    
    # Calculate histograms for each dataset
    histograms = []
    for counts in filtered_bb_count_list:
        hist, _ = np.histogram(counts, bins=bins)
        histograms.append(hist)
    
    x = np.arange(min_count, max_count + 1)
    
    # Set up colors and bar width
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    width = 0.15  # width of the bars
    positions = [x + i * width for i in range(len(bb_count_list))]
    
    # Plot 1: Percentage Distribution
    plt.figure(figsize=(15, 8))
    for i, (hist, name, color) in enumerate(zip(histograms, bb_count_names, colors)):
        # Convert to percentages
        total = sum(hist)
        if total > 0:
            hist_percent = (hist / total) * 100
        else:
            hist_percent = hist
        plt.bar(positions[i], hist_percent, width=width, label=name, color=color, align='center')
    
    plt.title('Basic Block Count Distribution (≤ 20 BBs) Across Different Datasets')
    plt.xlabel('Number of Basic Blocks per Function')
    plt.ylabel('Percentage of Functions (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(x + width * (len(bb_count_list) - 1) / 2, x)
    plt.savefig("figures/multi_dataset_bb_dist_percent.pdf")
    plt.close()
    
    # Plot 2: Actual Counts
    plt.figure(figsize=(15, 8))
    for i, (hist, name, color) in enumerate(zip(histograms, bb_count_names, colors)):
        plt.bar(positions[i], hist, width=width, label=name, color=color, align='center')
    
    plt.title('Basic Block Count Distribution (≤ 20 BBs) - Actual Counts')
    plt.xlabel('Number of Basic Blocks per Function')
    plt.ylabel('Number of Functions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(x + width * (len(bb_count_list) - 1) / 2, x)
    plt.savefig("figures/multi_dataset_bb_dist_count.pdf")
    plt.close()


def main(eval_data_path ='/home/xiachunwei/slade_artifact_cgo24_second/eval_data/exebench',
         num_proc = 80):
    valid_synth = load_dataset(eval_data_path, split='valid_synth')
    test_synth = load_dataset(eval_data_path, split='test_synth')
    valid_synth = prepare_dataset(valid_synth, num_proc)
    # valid_synth.save_to_disk("valid_synth_filtered.json")
    test_synth = prepare_dataset(test_synth, num_proc)

    valid_synth_has_struct = valid_synth.filter(filter_defined_struct, num_proc=num_proc)
    test_synth_has_struct = test_synth.filter(filter_defined_struct, num_proc=num_proc)
    print("len(valid_synth_has_struct)", len(valid_synth_has_struct))
    print("len(test_synth_has_struct)", len(test_synth_has_struct))
    valid_synth_struct_all_access = valid_synth.filter(filter_not_all_struct_field_access, num_proc=num_proc)
    test_synth_struct_all_access = test_synth.filter(filter_not_all_struct_field_access, num_proc=num_proc)
    print("len(valid_synth_struct_all_access)", len(valid_synth_struct_all_access))
    print("len(test_synth_struct_all_access)", len(test_synth_struct_all_access))
    exit(0)
    # for record in valid_synth_has_struct:
    #     print(record["llvm_ir"]['code'][-1])
    #     print("////////////")
    #     print(record['func_def'])
    #     print("\n")
    
    # test_synth.save_to_disk("test_synth_filtered.json")
    thresholds = [5, 10, 20, 50]
    print(f"Valid Synthesis Set:")
    # for threshold in thresholds:
    #     count_records_with_min_bbs(valid_synth, threshold)
    # valid_synth = valid_synth.filter(filter_func_with_min_bbs, num_proc=num_proc)
    # Plot distributions for both datasets
    # plot_bb_distribution(valid_synth, "Validation Set", "figures/valid_bb_dist.png")
    print(f"Test Synthesis Set:")
    # for threshold in thresholds:
    #     count_records_with_min_bbs(test_synth, threshold)
    # test_synth = test_synth.filter(filter_func_with_min_bbs, num_proc=num_proc)
    # plot_bb_distribution(test_synth, "Test Set", "figures/test_bb_dist.png")
    # plot_combined_bb_distribution(valid_synth, test_synth, "figures/combined_bb_dist.eps")

    valid_bb_counts = []
    test_bb_counts = []

    # Collect counts for valid set
    # count = 0
    # for record in valid_synth:
    #     if record['func_info'] and 'functions' in record['func_info']:
    #         if len(record['func_info']['functions']) == 0:
    #             print(record['func_info'])
    #             with open("/tmp/tmp.ll", 'w') as f:
    #                 f.write(record["llvm_ir"]['code'][-1])
    #             cmd = ["/home/xiachunwei/Software/llvm-project/mybuilddir/bin/llvm-parser-checker", "/tmp/tmp.ll"]
    #             ret = subprocess.run(cmd, capture_output=True, text=True)
    #             print(ret.stderr)
    #             print(ret.stdout)
    #             print(record['func_def'])
    #         for func in record['func_info']['functions']:
    #             count += 1
    #             valid_bb_counts.append(len(func['bbcount']))
    #     else:
    #         print(record['func_info'])
    # print(count)
    # # Collect counts for test set
    # for record in test_synth:
    #     if record['func_info'] and 'functions' in record['func_info']:
    #         for func in record['func_info']['functions']:
    #             test_bb_counts.append(len(func['bbcount']))

    redis_list = read_bb_from_project("/home/xiachunwei/Software/redis/redis_functions_info.pkl")
    coreutils_list = read_bb_from_project("/home/xiachunwei/Software/coreutils/coreutils_functions_info.pkl")
    ffmpeg_list = read_bb_from_project("/home/xiachunwei/Software/FFmpeg/redis_functions_info.pkl")
    plot_bb_for_a_series_of_bbcounts(
        [valid_bb_counts, test_bb_counts, redis_list, coreutils_list, ffmpeg_list],
        ["exebench-valid", "exebench-test", "redis", "coreutils", "ffmpeg"]
    )
    # filter the valid_synth and test_synth based on the basic block count
    # valid_synth_filtered = filter_by_llvm_diff(valid_synth)
    # test_synth_filtered = filter_by_llvm_diff(test_synth)
    # print("before: {}".format(len(valid_synth)))
    # print("after: {}".format(len(valid_synth_filtered)))
    # print("before: {}".format(len(test_synth)))
    # print("after: {}".format(len(test_synth_filtered)))
    # valid_synth_filtered.save_to_disk("valid_synth_filtered_bb_filtered.json")
    # test_synth_filtered.save_to_disk("test_synth_filtered_bb_filtered.json")
    print("len(valid_synth_has_struct)", len(valid_synth_has_struct))
    print("len(test_synth_has_struct)", len(test_synth_has_struct))
    print("len(valid_synth_struct_all_access)", len(valid_synth_struct_all_access))
    print("len(valid_synth_struct_all_access)", len(test_synth_struct_all_access))

if __name__ == "__main__":
    fire.Fire(main)
    