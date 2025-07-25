
import fire
from datasets import load_from_disk
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly

def main(idx = 93,
        dataset_dir_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_ir/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100",
         output_dir = "validation/deepseek-r1/deepseek-r1-assembly-with-comments/"):
    predict = ""
    with open(f"validation/deepseek-r1/deepseek-r1-assembly-with-comments/sample_{idx}/corrected_llvm_ir.ll", 'r') as f:
        predict = f.read()
    dataset = load_from_disk(dataset_dir_path)
    sample_dir = output_dir + f"/sample_{idx}"
    record = dataset[idx]
    predict_execution_success, target_execution_success = False, False
    predict_compile_success, predict_assembly_path = compile_llvm_ir(predict, sample_dir, name_hint="corrected_llvm_ir")
    if predict_compile_success:
        with open(predict_assembly_path, 'r') as f:
            predict_execution_success = eval_assembly(record, f.read())
    target_compile_success, target_assembly_path = compile_llvm_ir(record["llvm_ir"]["code"][-1], sample_dir, name_hint="target")
    if target_compile_success:
        with open(target_assembly_path, 'r') as f:
            target_execution_success = eval_assembly(record, f.read())

    validation_results = {
        "idx": idx,
        "path": record["path"],
        "func_head": record["func_head"],
        "predict_compile_success": predict_compile_success,
        "predict_execution_success": predict_execution_success,
        "target_compile_success": target_compile_success,
        "target_execution_success": target_execution_success,
    }
    print(validation_results)


if __name__ == "__main__":
    fire.Fire(main)