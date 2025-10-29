import os
import pickle
import datasets
from multiprocessing import Pool
from pathlib import Path
from datasets import load_from_disk
from qdrant_client import QdrantClient

from models.gemini.llm_decompiler import LLMDecompileRecord
from models.gemini.llm_decompiler import PromptType
from models.rag.embedding_client import RemoteEmbeddingModel

embedding_url = "http://localhost:8001"
HOME_DIR = os.path.expanduser("~")
qdrant_host = "localhost"
qdrant_port = 6333
dataset_for_qdrant_dir = f"{HOME_DIR}/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"

# args = [(record, idx, output_dir, model_name, prompt_type, remove_comments, num_generate) for idx, record in enumerate(dataset)]
#     with Pool(processes=num_processes) as pool:
#         results = pool.starmap(decompile_func, args)
def evaluate_record(args):
    record, idx, client, model_name, num_generate, output_dir, prompt_type, remove_comments, num_retry, embedding_model, qdrant_client, dataset_for_qdrant_dir = args
    llm_decompile_record = LLMDecompileRecord(record, idx, client,
                                                model_name, num_generate,
                                                output_dir, prompt_type,
                                                remove_comments, num_retry)
    llm_decompile_record.evaluate_existing_output()
    llm_decompile_record.llm_client = None
    return llm_decompile_record

def evaluate_exebenc_existing_output(dataset: list,
                                     output_dir: str,
                                     prompt_type,
                                     num_generate: int = 8,
                                     remove_comments: bool = True,
                                     num_retry: int = 10):
    llm_decompile_record_list = []
    client = None
    model_name = None
    embedding_model = RemoteEmbeddingModel(embedding_url)
    # Setup Qdrant
    # qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    qdrant_client = None
    args = [(record, idx, client, model_name, num_generate, output_dir, prompt_type, remove_comments, num_retry, embedding_model, qdrant_client, dataset_for_qdrant_dir) 
            for idx, record in enumerate(dataset)]
    
    with Pool(processes=64) as pool:
        llm_decompile_record_list = pool.map(evaluate_record, args)
        
    return llm_decompile_record_list


if __name__ == "__main__":
    remove_comments = True
    with_comments = "without" if remove_comments else "with"
    prompt_type = PromptType.SIMILAR_RECORD
    model = "Qwen3-32B"
    dataset_paires = [
        (
            f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_without_loops_164", 
            # f"{HOME_DIR}/Projects/validation/{model}/sample_without_loops_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
            "/data1/xiachunwei/Projects/validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-without-comments"
        ),
        (
            f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_164", 
            # f"{HOME_DIR}/Projects/validation/{model}/sample_loops_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
            "/data1/xiachunwei/Projects/validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-without-comments"
        ),
        (
            f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
            # f"{HOME_DIR}/Projects/validation/{model}/sample_only_one_bb_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
            "/data1/xiachunwei/Projects/validation/Qwen3-32B/sample_only_one_bb_Qwen3-32B-n8-assembly-without-comments"
        ),
    ]
    for dataset_dir_path, response_output_dir in dataset_paires:
        dataset = load_from_disk(dataset_dir_path)
        llm_decompile_record_list = evaluate_exebenc_existing_output(dataset, response_output_dir, prompt_type, num_generate=8, remove_comments=remove_comments)
        pickle.dump(llm_decompile_record_list, open(os.path.join(response_output_dir, "llm_decompile_results.pkl"), "wb"))
        