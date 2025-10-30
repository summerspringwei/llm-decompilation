import os
import pickle
import argparse
from multiprocessing import Pool
import faulthandler, signal

from openai import OpenAI
from datasets import load_from_disk, Dataset
from qdrant_client import QdrantClient

from utils.mylogger import logger
from models.gemini.llm_decompiler import LLMDecompileRecord, PromptType
from models.rag.embedding_client import RemoteEmbeddingModel

HOME_DIR = os.path.expanduser("~")
# Service config: Key: model name, Value: (client, model_name)

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run decompilation with specified model')
    parser.add_argument('--model',
                        type=str,
                        default="Qwen3-32B",
                        help='Model to use for decompilation')

    parser.add_argument('--host',
                        type=str,
                        default="localhost",
                        help='Host to use for decompilation')

    parser.add_argument('--port',
                        type=str,
                        default="9001",
                        help='Port to use for decompilation')

    parser.add_argument('--qdrant_host',
                        type=str,
                        default="localhost",
                        help='Host to use for Qdrant')

    parser.add_argument('--qdrant_port',
                        type=str,
                        default="6333",
                        help='Port to use for Qdrant')

    parser.add_argument('--embedding_model',
                        type=str,
                        default="http://localhost:8001/v1",
                        help='Embedding model to use for RAG')

    parser.add_argument('--prompt-type',
                        type=str,
                        default="ghidra-decompile",
                        help='Prompt type to use for decompilation')
    args = parser.parse_args()
    return args

args = parse_args()
model = args.model
host = args.host
port = args.port
# embedding_model = RemoteEmbeddingModel(args.embedding_model)

embedding_client = OpenAI(api_key="abcd", base_url=args.embedding_model)
prompt_type = PromptType(args.prompt_type)
qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
dataset_for_qdrant_dir = f"{HOME_DIR}/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"

SERVICE_CONFIG = {
    "Qwen3-32B": (OpenAI(
        api_key="token-llm4decompilation-abc123",
        base_url=f"http://{host}:{port}/v1",
    ), "Qwen3-32B"),
    "Qwen3-30B-A3B": (OpenAI(
        api_key="token-llm4decompilation-abc123",
        base_url=f"http://{host}:{port}/v1",
    ), "Qwen3-30B-A3B"),
    "Huoshan-DeepSeek-R1": (OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=1800), "ep-20250317013717-m9ksl"),
    "OpenAI-GPT-4.1": (OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), ),
                       "gpt-4.1")
}

# Set client and model here
global client
client, model_name = SERVICE_CONFIG[model]


def decompile_func(record, idx, output_dir, model_name: str, prompt_type: PromptType, remove_comments: bool = True, num_generate: int = 8):
    faulthandler.register(signal.SIGUSR1)
    llm_decompile_record = LLMDecompileRecord(record, idx, client, model_name, num_generate, output_dir, prompt_type, remove_comments, embedding_client=embedding_client, embedding_model_name="Qwen3-Embedding-8B")
    llm_decompile_record.get_initial_prompt()
    llm_decompile_record.decompile_and_evaluate(llm_decompile_record.initial_prompt, -1)
    llm_decompile_record.correct_one()
    # Remove unpickleable objects before returning for multiprocessing
    llm_decompile_record.llm_client = None
    llm_decompile_record.embedding_client = None
    return llm_decompile_record


def LLM_predict_openai_api(dataset: list,
                           output_dir: str,
                           prompt_type: PromptType,
                           num_processes: int = 8,
                           remove_comments: bool = True,
                           num_generate: int = 8):
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    # We need to save the similar records for in-context learning
    if not os.path.exists(os.path.join(output_dir, "similar_records")):
        os.makedirs(os.path.join(output_dir, "similar_records"), exist_ok=True)
    args = [(record, idx, output_dir, model_name, prompt_type, remove_comments, num_generate) for idx, record in enumerate(dataset)]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(decompile_func, args)
    pickle.dump(results, open(os.path.join(output_dir, "results.pkl"), "wb"))


    # Show the summary of the results
    predict_compile_success_count = 0
    predict_execution_success_count = 0
    target_compile_success_count = 0
    target_execution_success_count = 0
    for result in results:
        predict_compile_success, predict_execution_success = result.predict_has_compile_and_execution_sucess()
        if predict_compile_success:
            predict_compile_success_count += 1
        if predict_execution_success:
            predict_execution_success_count += 1
        target_compile_success, target_execution_success = result.target_has_compile_and_execution_sucess()
        if target_compile_success:
            target_compile_success_count += 1
        if target_execution_success:
            target_execution_success_count += 1
    print(f"Number of predict_compile_success: {predict_compile_success_count}")
    print(f"Number of predict_execution_success: {predict_execution_success_count}")
    print(f"Number of target_compile_success: {target_compile_success_count}")
    print(f"Number of target_execution_success: {target_execution_success_count}")



def main(dataset_dir_path,
         output_dir,
         num_processes: int = 1,
         num_generate: int = 8,
         remove_comments: bool = True,
         prompt_type: PromptType = PromptType.GHIDRA_DECOMPILE,
         num_retry: int = 10):
    dataset = load_from_disk(dataset_dir_path)
    if not os.path.exists(response_output_dir):
        os.makedirs(response_output_dir, exist_ok=True)
    LLM_predict_openai_api(dataset, output_dir, prompt_type=prompt_type, num_processes=num_processes, remove_comments=remove_comments, num_generate=num_generate)
    # TODO(Chunwei) TODO: Add fix 



if __name__ == "__main__":
    remove_comments = True
    with_comments = "without" if remove_comments else "with"
    prompt_type = PromptType.GHIDRA_DECOMPILE
    dataset_paires = [
        # (
        #     f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
        #     f"{HOME_DIR}/Projects/validation/{model}/sample_only_one_bb_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
        # ),
        (
            f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_without_loops_164", 
            f"{HOME_DIR}/Projects/validation/{model}/sample_without_loops_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
        ),
        # (
        #     f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_164", 
        #     f"{HOME_DIR}/Projects/validation/{model}/sample_loops_{model}-n8-assembly-{with_comments}-comments-{prompt_type}"
        # ),
    ]
    for dataset_dir_path, response_output_dir in dataset_paires:
        if not os.path.exists(response_output_dir):
            os.makedirs(response_output_dir, exist_ok=True)
        main(dataset_dir_path, response_output_dir, num_processes=32, remove_comments=remove_comments, prompt_type=prompt_type, num_generate=8)
        
