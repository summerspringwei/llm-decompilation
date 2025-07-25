
from typing import Dict
import datasets


def extract_ir_and_assembly(row: Dict)->Dict :
    return {
        "input": row["asm"]["code"][-1],
        "output": row["llvm_ir"]["code"][0],
        "file": row["path"]
    }

def prepare_exebench_dataset(exebench_path: str, start: int = 0, end: int = 100):
    dataset = datasets.load_from_disk(exebench_path)
    selected_dataset = dataset.select(range(start, end))
    filtered_dataset = selected_dataset.filter(lambda x: len(x["llvm_ir"]["code"]) > 0)
    input_output_dataset = filtered_dataset.map(extract_ir_and_assembly, num_proc=40)
    # input_output_dataset = input_output_dataset.remove_columns(["asm", "llvm_ir", "fname",
    #                                                              "func_def", "func_head", "func_head_types", "path", "real_deps",
    #                                                              "real_exe_wrapper", "real_io_pairs", "real_iospec", "ref", "signature", "synth_deps", "synth_exe_wrapper",
    #                                                              "synth_io_pairs", "synth_iospec"])
    return input_output_dataset


if __name__ == "__main__":
    input_output_dataset = prepare_exebench_dataset("/data/xiachunwei/Datasets/train_synth_rich_io_filtered_llvm_ir/")
    # print(input_output_dataset[0])
    for k, v in input_output_dataset[0].items():
        print(k, v)