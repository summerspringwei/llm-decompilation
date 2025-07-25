import os
import pickle

from datasets import load_from_disk
from utils.openai_helper import extract_llvm_code_from_response

def get_failed_samples(
    dataset_dir,
    num_retry = 10,
    output_dir = "validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-with-comments",
    ):
    """
    Get the failed samples  extract_llvm_code_from_response
    """
    dataset = load_from_disk(dataset_dir)
    fix_evaluation_list = pickle.load(open(os.path.join(output_dir, "fix_evaluation_list.pkl"), "rb"))
    record_response_pair_list = []
    failed_samples = []
    for sample_evaluation in fix_evaluation_list:
        if len(sample_evaluation) > 0 and not any(sample_evaluation[-1]["predict_execution_success"]):
            failed_samples.append(sample_evaluation[-1]['idx'])
    
    for idx in failed_samples:
        response = pickle.load(open(os.path.join(output_dir, f"response_{idx}_retry_{num_retry}.pkl"), "rb"))
        record_response_pair_list.append((dataset[idx], response))
        sample_dir = os.path.join(output_dir, f"sample_{idx}_retry_{num_retry}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, f"target.ll"), "w") as f:
            f.write(dataset[idx]["llvm_ir"]['code'][-1])
        llvm_code_list = extract_llvm_code_from_response(response)
        compile_list = fix_evaluation_list[idx][num_retry-1]['predict_compile_success']
        print(idx, compile_list)
        for generation_idx, (llvm_code, compile_success) in enumerate(zip(llvm_code_list, compile_list)):
            if compile_success:
                if llvm_code != []:
                    with open(os.path.join(sample_dir, f"predict_retry_{generation_idx}.ll"), "w") as f:
                        f.write(llvm_code)
        print(sample_dir)
    print(failed_samples)
    return record_response_pair_list


if __name__ == "__main__":
    model = "Qwen3-32B"
    dataset_paires = [
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164", f"validation/{model}/sample_without_loops_{model}-n8-assembly-with-comments"),
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_164", f"validation/{model}/sample_loops_{model}-n8-assembly-with-comments"),
        ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
         f"validation/{model}/sample_only_one_bb_{model}-n8-assembly-with-comments"
         ),
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_on_bb_function_call", 
        #     f"validation/{model}/sample_one_bb_with_functions_{model}-n8-assembly-with-comments"),
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_on_bb_without_function_call", f"validation/{model}/sample_one_bb_wo_functions_{model}-n8-assembly-with-comments")
    ]
    record_response_pair_list = get_failed_samples(
        dataset_dir=dataset_paires[0][0],
        output_dir=dataset_paires[0][1],
        num_retry=10
    )
    