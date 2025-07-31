import os
import pickle
import logging
import argparse
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from datasets import load_from_disk, Dataset
from qdrant_client import QdrantClient

from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.preprocessing_assembly import preprocessing_assembly
from utils.preprocessing_llvm_ir import preprocessing_llvm_ir
from utils.openai_helper import extract_llvm_code_from_response, format_compile_error_prompt, format_execution_error_prompt
from models.rag.exebench_qdrant_base import load_embedding_model, find_similar_records_in_exebench_synth_rich_io
from utils.openai_helper import GENERAL_INIT_PROMPT, SIMILAR_RECORD_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


HOME_DIR = os.path.expanduser("~")
# Service config: Key: model name, Value: (client, model_name)


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
                    default="/data1/xiachunwei/Datasets/Models/Qwen3-Embedding-8B",
                    help='Embedding model to use for RAG')

parser.add_argument('--in-context-learning',
                    action='store_true',
                    default=True,
                    help='Use in context learning')

args = parser.parse_args()

model = args.model
host = args.host
port = args.port
embedding_model_path = args.embedding_model
in_context_learning = args.in_context_learning

qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
dataset_for_qdrant_dir = f"{HOME_DIR}/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"

SERVICE_CONFIG = {
    "Qwen3-32B": (OpenAI(
        api_key="token-llm4decompilation-abc123",
        base_url=f"http://{host}:{port}/v1",
    ), "Qwen3-32B", GENERAL_INIT_PROMPT),
    "Qwen3-30B-A3B": (OpenAI(
        api_key="token-llm4decompilation-abc123",
        base_url=f"http://{host}:{port}/v1",
    ), "Qwen3-30B-A3B", GENERAL_INIT_PROMPT),
    "Huoshan-DeepSeek-R1": (OpenAI(api_key=os.environ.get("ARK_STREAM_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=1800), "ep-20250317013717-m9ksl", GENERAL_INIT_PROMPT),
    "OpenAI-GPT-4.1": (OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), ),
                       "gpt-4.1", GENERAL_INIT_PROMPT)
}

# Set client and model here
global client
client, model_name, input_prompt = SERVICE_CONFIG[model]


def huoshan_deepseek_r1(client, prompt: str, n: int = 8):
    response = client.chat.completions.create(model=model_name,
                                              messages=[
                                                  {
                                                      "role": "user",
                                                      "content": prompt
                                                  },
                                              ],
                                              n=n,
                                              stream=False,
                                              timeout=7200)
    return response


def huoshan_deepseek_r1_batch(client, prompt_list: list[str]):
    response = client.chat.completions.create(model='ep-20250317013717-m9ksl',
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              } for prompt in prompt_list],
                                              stream=False)
    return response


def prepare_prompt(record, remove_comments: bool = True, embedding_model=None):
    asm_code = record["asm"]["code"][-1]
    asm_code = preprocessing_assembly(asm_code,
                                      remove_comments=remove_comments)
    prompt = input_prompt.format(asm_code=asm_code)
    return prompt


def prepare_prompt_from_similar_record(record, remove_comments: bool = True, embedding_model=None, idx: int = 0, output_dir: str = ""):
    similar_record, score = find_similar_records_in_exebench_synth_rich_io(qdrant_client, embedding_model, record, dataset_for_qdrant_dir)
    asm_code = record["asm"]["code"][-1]
    asm_code = preprocessing_assembly(asm_code,
                                      remove_comments=remove_comments)
    similar_asm_code = similar_record["asm"]["code"][-1]
    similar_asm_code = preprocessing_assembly(similar_asm_code,
                                              remove_comments=remove_comments)
    similar_llvm_ir = similar_record["llvm_ir"]["code"][-1]
    similar_llvm_ir = preprocessing_llvm_ir(similar_llvm_ir)
    prompt = SIMILAR_RECORD_PROMPT.format(asm_code=asm_code, similar_asm_code=similar_asm_code, similar_llvm_ir=similar_llvm_ir)
    pickle.dump(similar_record, open(os.path.join(output_dir, f"similar_record_{idx}.pkl"), "wb"))
    return prompt


prepare_prompt_func = prepare_prompt_from_similar_record if in_context_learning else prepare_prompt


class EvaluationResult:
    def __init__(self, predict_compile_success, predict_execution_success, error_msg):
        self.predict_compile_success = predict_compile_success
        self.predict_execution_success = predict_execution_success
        self.error_msg = error_msg


def evaluate_response(response, record, idx, validate_dir):
    """
    Evaluate the response from the LLM and validate the LLVM IR prediction.
    Args:
        response (ChatCompletion): The response from the LLM.
        record (dict): The record from the dataset.
        idx (int): The index of the record.
        validate_dir (str): The directory to save the validation results.
    Returns:
        dict: A dictionary containing the validation results.
    """
    predict_list = extract_llvm_code_from_response(response)
    sample_dir = os.path.join(validate_dir, f"sample_{idx}")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
    predict_compile_success_list = []
    predict_execution_success_list = []

    def eval_predict(predict):
        predict_execution_success = False
        predict_compile_success, predict_assembly_path, error_msg = compile_llvm_ir(
            predict, sample_dir, name_hint="predict")
        if predict_compile_success:
            with open(predict_assembly_path, 'r') as f:
                predict_execution_success = eval_assembly(record, f.read())
        return EvaluationResult(predict_compile_success, predict_execution_success, error_msg)

    for predict in predict_list:
        if isinstance(predict, list) and len(predict) == 0:
            logging.error(f"Empty predict in predict_list: {predict_list}")
    # TODO(Chunwei) Now the eval_predict and compile_llvm_ir functions are not thread safe
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(eval_predict, predict_list))

    predict_compile_success_list = [r.predict_compile_success for r in results]
    predict_execution_success_list = [r.predict_execution_success for r in results]
    error_msg_list = [r.error_msg for r in results]

    target_compile_success, target_assembly_path, target_error_msg = compile_llvm_ir(
        record["llvm_ir"]["code"][-1], sample_dir, name_hint="target")
    if target_compile_success:
        with open(target_assembly_path, 'r') as f:
            target_execution_success = eval_assembly(record, f.read())

    validation_results = {
        "idx": idx,
        "path": record["path"],
        "func_head": record["func_head"],
        "predict_compile_success": predict_compile_success_list,
        "predict_execution_success": predict_execution_success_list,
        "predict_error_msg": error_msg_list,
        "target_compile_success": target_compile_success,
        "target_execution_success": target_execution_success,
    }

    return validation_results


def process_one(prompt: str, idx: int, output_dir: str,
                record: dict) -> tuple[ChatCompletion, dict]:
    """Call the openai api to inference one sample and evaluate the sample

    """
    response_file_path = os.path.join(output_dir, f"response_{idx}.pkl")
    prompt_dir = os.path.join(output_dir, "prompts")
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, f"prompt_{idx}.txt"), "w") as f:
        f.write(prompt)
    if os.path.exists(response_file_path):
        response = pickle.load(open(response_file_path, "rb"))
    else:
        response = huoshan_deepseek_r1(client, prompt)
        # Make sure first save result to persistant storage
        pickle.dump(response, open(response_file_path, "wb"))
    # validate the output
    validation_results = {}
    try:
        validation_results = evaluate_response(response, record, idx,
                                               output_dir)
    except Exception as e:
        logging.warning(f"Error in evaluating response for index {idx}: {e}")
    return response, validation_results


def LLM_predict_openai_api(dataset: list,
                           output_dir: str,
                           num_processes: int = 8,
                           remove_comments: bool = True):
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    # We need to save the similar records for in-context learning
    if not os.path.exists(os.path.join(output_dir, "similar_records")):
        os.makedirs(os.path.join(output_dir, "similar_records"), exist_ok=True)
    embedding_model = None
    if in_context_learning:
        embedding_model = load_embedding_model(embedding_model_path)
    args = [(prepare_prompt_func(record,
                                remove_comments, embedding_model, idx, os.path.join(output_dir, "similar_records")), idx, output_dir, record)
                for idx, record in enumerate(dataset)]
    if in_context_learning:
        del embedding_model
        print("del embedding model")
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_one, args)
        pickle.dump(results, open(os.path.join(output_dir, "results.pkl"),
                                  "wb"))

    compile_success_count = 0
    execution_success_count = 0
    target_execution_success_count = 0
    for _, validation_results in results:
        if "predict_compile_success" not in validation_results.keys() or \
            "predict_execution_success" not in validation_results.keys() or \
            "target_execution_success" not in validation_results.keys():
            print("GGG", validation_results)
            continue
        compile_success_count += any(
            validation_results["predict_compile_success"])
        execution_success_count += any(
            validation_results["predict_execution_success"])
        target_execution_success_count += validation_results[
            "target_execution_success"]
    print(f"Number of predict_compile_success: {compile_success_count}")
    print(f"Number of predict_execution_success: {execution_success_count}")
    print(
        f"Number of target_execution_success: {target_execution_success_count}"
    )


def prepare_fix_prompt(record, chat_response, validation_result, idx,
                       retry_count, output_dir, remove_comments: bool = True):
    predict_list = extract_llvm_code_from_response(chat_response)
    sample_dir = os.path.join(output_dir, f"sample_{idx}_retry_{retry_count}")
    # Get the similar record for in-context learning
    similar_record_dir = os.path.join(output_dir, "similar_records")
    similar_record_path = os.path.join(similar_record_dir, f"similar_record_{idx}.pkl")
    similar_record = pickle.load(open(similar_record_path, "rb"))
    similar_asm_code = similar_record["asm"]["code"][-1]
    similar_asm_code = preprocessing_assembly(similar_asm_code,
                                              remove_comments=remove_comments)
    similar_llvm_ir = similar_record["llvm_ir"]["code"][-1]
    similar_llvm_ir = preprocessing_llvm_ir(similar_llvm_ir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
    predict_compile_success_list = validation_result["predict_compile_success"]
    target_asm_code = record["asm"]["code"][-1]
    target_asm_code = preprocessing_assembly(target_asm_code,
                                              remove_comments=remove_comments)
    error_msg, fix_idx = None, 0
    # 1. If there is no choice that is compile success, we choose the one that is compile success
    if not any(predict_compile_success_list):
        error_msg_list = validation_result['predict_error_msg']
        predict = ""
        for choice_idx, error_msg in enumerate(error_msg_list):
            if error_msg is not None and error_msg.strip() != "":
                error_msg = error_msg.strip()
                fix_idx = choice_idx
                predict = predict_list[fix_idx]
                if isinstance(predict, list) and len(predict) > 0:
                    predict = predict[0]
                    break
        if predict == "":
            predict = predict_list[0]
        prompt = format_compile_error_prompt(target_asm_code,
                                             predict, error_msg, in_context_learning, similar_asm_code, similar_llvm_ir)
    # 2 If there is one choice that is compile success, we choose the one that is compile success
    else:
        for choice_idx, predict in enumerate(predict_compile_success_list):
            if predict:
                fix_idx = choice_idx
                break
        predict = predict_list[fix_idx]
        # Get the assembly code of the predict
        success, assembly_path, error_msg = compile_llvm_ir(
            predict, sample_dir, name_hint="predict")
        assert success, f"Failed to compile the predict for index {idx} with error message: {error_msg}"
        with open(assembly_path, 'r') as f:
            predict_assembly = f.read()
            predict_assembly = preprocessing_assembly(predict_assembly,
                                                      remove_comments=remove_comments)
        prompt = format_execution_error_prompt(target_asm_code,
                                               predict, predict_assembly, in_context_learning, similar_asm_code, similar_llvm_ir)
    return prompt


def correct_one(chat_response,
                idx: int,
                record: dict,
                output_dir: str,
                validation_result: dict,
                num_retry: int = 10,
                num_generate: int = 8,
                remove_comments: bool = True) -> bool:
    if any(validation_result["predict_execution_success"]):
        print(
            f"Index: {idx}, {validation_result['predict_compile_success']}, {validation_result['predict_execution_success']}"
        )
        return True, {}
    prompt_dir = os.path.join(output_dir, "prompts")
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir, exist_ok=True)
    predict_list = extract_llvm_code_from_response(chat_response)
    sample_dir = os.path.join(output_dir, f"sample_{idx}")
    count = 0
    fix_success = False
    predict_execution_success_list = validation_result[
        "predict_execution_success"]
    fix_evaluation_list = []
    while count < num_retry and (not any(predict_execution_success_list)):
        count += 1
        logger.info(f"Retrying {count} times for index {idx}")
        # 1. Prepare the prompt for the next retry
        prompt = prepare_fix_prompt(record, chat_response, validation_result,
                                    idx, count, output_dir, remove_comments)
        with open(os.path.join(prompt_dir, f"prompt{idx}_retry_{count}.txt"), "w") as f:
            f.write(prompt)
        # 2. Call the LLM to fix the error
        pkl_file_path = os.path.join(output_dir, f"response_{idx}_retry_{count}.pkl")
        if os.path.exists(pkl_file_path):
            response = pickle.load(open(pkl_file_path, "rb"))
        else:
            response = huoshan_deepseek_r1(client, prompt, num_generate)
            pickle.dump(
                response,
                open(pkl_file_path,
                    "wb"))
        # 3. Evaluate the response
        validation_result = evaluate_response(response, record, idx,
                                              output_dir)
        fix_evaluation_list.append(validation_result)
        # 4. If the response is not correct, we need to fix the error
        if not any(validation_result["predict_execution_success"]):
            chat_response = response
            continue
        # 5. If the response is correct, we save the correct LLVM IR
        else:
            predict_list = extract_llvm_code_from_response(chat_response)
            for execution_sucess, predict in zip(
                    validation_result["predict_execution_success"],
                    predict_list):
                if execution_sucess and (predict is not None) and (predict != ""):
                    if isinstance(predict, list) and len(predict) == 0:
                        logger.warning(f"Empty predict for index {idx}")
                        continue
                    with open(os.path.join(sample_dir, "corrected_llvm_ir.ll"),
                              'w') as f:
                        f.write(predict)
                        fix_success = True
                        logger.info(
                            f"Corrected LLVM IR saved to {os.path.join(sample_dir, 'corrected_llvm_ir.ll')}"
                        )
                    break
            break
    if not fix_success:
        logger.warning(
            f"Failed to correct LLVM IR for index {idx} after {num_retry} retries."
        )
    return fix_success, fix_evaluation_list


def fix_all(dataset, dir_path, num_processes=8, num_retry=10, num_generate=8, remove_comments: bool = True):
    response = pickle.load(open(os.path.join(dir_path, "results.pkl"), "rb"))
    fix_count = 0
    args_list = [(chat, idx, dataset[idx], dir_path, validation, num_retry,
                num_generate, remove_comments)
                 for idx, (chat, validation) in enumerate(response)]
    # args_list = args_list[:8]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(correct_one, args_list)
    fix_count = sum([r[0] for r in results])
    fix_evaluation_list = [r[1] for r in results]
    print("Total passed: ", fix_count)
    pickle.dump(fix_evaluation_list,
                open(os.path.join(dir_path, "fix_evaluation_list.pkl"), "wb"))
    return fix_count, fix_evaluation_list


def main(dataset_dir_path,
         output_dir,
         num_processes=1,
         num_retry=10,
         num_generate=8,
         remove_comments: bool = True):
    dataset = load_from_disk(dataset_dir_path)
    if not os.path.exists(response_output_dir):
        os.makedirs(response_output_dir, exist_ok=True)
    LLM_predict_openai_api(dataset, output_dir, num_processes=num_processes, remove_comments=remove_comments)
    fix_all(dataset,
            output_dir,
            num_processes=num_processes,
            num_retry=num_retry,
            num_generate=num_generate,
            remove_comments=remove_comments)


if __name__ == "__main__":
    remove_comments = True
    with_comments = "without" if remove_comments else "with"
    in_context_learning_str = "in-context-learning" if in_context_learning else "no-in-context-learning"
    dataset_paires = [
        (
            f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_without_loops_164", 
            f"{HOME_DIR}/Projects/validation/{model}/sample_without_loops_{model}-n8-assembly-{with_comments}-comments-{in_context_learning_str}"
        ),
        # (
        #     f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_164", 
        #     f"{HOME_DIR}/Projects/validation/{model}/sample_loops_{model}-n8-assembly-{with_comments}-comments-{in_context_learning_str}"
        # ),
        # (
        #     f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
        #     f"{HOME_DIR}/Projects/validation/{model}/sample_only_one_bb_{model}-n8-assembly-{with_comments}-comments-{in_context_learning_str}"
        # ),
    ]
    for dataset_dir_path, response_output_dir in dataset_paires:
        if not os.path.exists(response_output_dir):
            os.makedirs(response_output_dir, exist_ok=True)
        main(dataset_dir_path, response_output_dir, num_processes=64, remove_comments=remove_comments)
        
