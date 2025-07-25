from datasets import load_from_disk
from utils.func_info import FuncInfo
from models.gemini.prepare_hard_dataset import filter_cannot_parse, map_func_info


def func_info_similar(func_info_1: FuncInfo, func_info_2: FuncInfo, max_diff_bb_count: int = 2) -> bool:
    """Check if two function infos are similar"""
    # Called functions:
    if (len(func_info_1.called_functions) == 0) != (len(func_info_2.called_functions) == 0):
        return False
    # Loops:
    if (func_info_1.num_loops == 0) != (func_info_2.num_loops == 0):
        return False 
    # For basic blocks, only one basic block:
    if (len(func_info_1.bb_count_list) == 1) != (len(func_info_2.bb_count_list) == 1):
        return False
    # For basic blocks, the difference in number of basic blocks is not too large:
    if abs(len(func_info_1.bb_count_list) - len(func_info_2.bb_count_list)) > max_diff_bb_count:
        return False
    return True


def find_similar_code(record: dict, dataset: list[dict]) -> dict:
    """Based on the function info in record, we find a similar function in dataset and return the record
    Args:
        record: The record to find similar code for
        dataset: The dataset to find similar code in
    Returns:
        The record with the similar code
    """
    info1 = FuncInfo(record)
    
    for item in dataset:
        info2 = FuncInfo(item)
        if func_info_similar(info1, info2):
            return item
    return None


def test_func_info_similar():
    dataset_164 = load_from_disk("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_164")
    dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"
    dataset = load_from_disk(dataset_path)
    dataset = dataset.filter(filter_cannot_parse, num_proc=16, load_from_cache_file=True)
    dataset = dataset.map(map_func_info, num_proc=16, load_from_cache_file=True)
    for record in dataset_164:
        similar_record = find_similar_code(record, dataset)
        if similar_record is None:
            print("not similar")
            print(record["func_def"])
            print("=="*100)
        else:
            print("similar")
            print(record["llvm_ir"]["code"][-1])
            print("#" * 100)
            print(similar_record["llvm_ir"]["code"][-1])
            print("=="*100)
        


if __name__ == "__main__":
    test_func_info_similar()
