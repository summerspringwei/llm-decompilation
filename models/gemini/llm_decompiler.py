import os
import pickle
import logging
import torch
import numpy as np
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from openai.types.chat.chat_completion import ChatCompletion

from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.preprocessing_assembly import preprocessing_assembly
from utils.preprocessing_llvm_ir import preprocessing_llvm_ir
from utils.openai_helper import extract_llvm_code_from_response, format_compile_error_prompt, format_execution_error_prompt
from models.rag.exebench_qdrant_base import find_similar_records_in_exebench_synth_rich_io
from utils.openai_helper import GENERAL_INIT_PROMPT, SIMILAR_RECORD_PROMPT, GHIDRA_DECOMPILE_TEMPLATE
from models.ghidra_decompile.ghidra_decompile_exebench import ghdria_decompile_record

from utils.mylogger import logger


class EvaluationResult:
    def __init__(self, compile_success: bool, execution_success: bool, error_msg: str, llvm_ir: str = None, assembly: str = None):
        self.compile_success = compile_success
        self.execution_success = execution_success
        self.error_msg = error_msg
        self.llvm_ir = llvm_ir
        self.assembly = assembly


class ResponseValidation:

    def __init__(self, prompt: str, response: ChatCompletion, retry_count: int,
                 predict_evaluation_results_list: list[EvaluationResult],
                 target_evaluation_result: EvaluationResult):
        self.prompt = prompt
        self.response = response
        self.retry_count = retry_count
        self.predict_evaluation_results_list = predict_evaluation_results_list
        self.target_evaluation_result = target_evaluation_result
        

    def get_num_execution_success(self) -> int:
        return sum(1 for result in self.predict_evaluation_results_list if result.execution_success)

    def get_num_compile_success(self) -> int:
        return sum(1 for result in self.predict_evaluation_results_list if result.compile_success)
    
    def get_first_compile_success_evaluation_result(self) -> EvaluationResult:
        return next((result for result in self.predict_evaluation_results_list if result.compile_success), self.predict_evaluation_results_list[0])

    def get_num_total(self) -> int:
        return len(self.predict_evaluation_results_list)
    
    def dump_correct_llvm_ir(self, output_dir: str):
        for result in self.predict_evaluation_results_list:
            if result.execution_success:
                with open(os.path.join(output_dir, f"correct_llvm_ir.ll"), 'w') as f:
                    f.write(result.llvm_ir)
                break


class PromptType(Enum):
    BASIC = "basic"
    SIMILAR_RECORD = "in-context-learning"
    GHIDRA_DECOMPILE = "ghidra-decompile"
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class LLMDecompileRecord:

    def __init__(self,
                 record,
                 idx,
                 llm_client,
                 model_name: str,
                 num_generate: int,
                 output_dir: str,
                 prompt_type: PromptType,
                 remove_comments: bool = True,
                 num_retry: int = 10,
                 embedding_model=None,
                 qdrant_client=None,
                 dataset_for_qdrant_dir=None
                 ):
        self.record = record
        self.idx = idx
        self.llm_client = llm_client
        self.model_name = model_name
        self.num_generate = num_generate
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.remove_comments = remove_comments
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.dataset_for_qdrant_dir = dataset_for_qdrant_dir
        self.num_retry = num_retry
        self.retry_response_validation: dict[int, ResponseValidation] = {}
        self.similar_record = None
        self.score = None
        self.initial_prompt = None
        # self.initial_prompt = self.get_initial_prompt()

    def prepare_basic_prompt(self):
        asm_code = self.record["asm"]["code"][-1]
        asm_code = preprocessing_assembly(asm_code,
                                          remove_comments=self.remove_comments)
        prompt = GENERAL_INIT_PROMPT.format(asm_code=asm_code)
        return prompt

    # TODO(Chunwei) Need to use api to call embedding model to find the similar record
    def prepare_prompt_from_similar_record(self):
        self.similar_record, self.score = find_similar_records_in_exebench_synth_rich_io(
            self.qdrant_client, self.embedding_model, self.record,
            self.dataset_for_qdrant_dir)
        asm_code = self.record["asm"]["code"][-1]
        asm_code = preprocessing_assembly(asm_code,
                                          remove_comments=self.remove_comments)
        similar_asm_code = self.similar_record["asm"]["code"][-1]
        similar_asm_code = preprocessing_assembly(
            similar_asm_code, remove_comments=self.remove_comments)
        similar_llvm_ir = self.similar_record["llvm_ir"]["code"][-1]
        similar_llvm_ir = preprocessing_llvm_ir(similar_llvm_ir)
        prompt = SIMILAR_RECORD_PROMPT.format(
            asm_code=asm_code,
            similar_asm_code=similar_asm_code,
            similar_llvm_ir=similar_llvm_ir)
        pickle.dump(
            self.similar_record,
            open(
                os.path.join(self.output_dir,
                             f"similar_record_{self.idx}.pkl"), "wb"))
        self.initial_prompt = prompt
        return prompt

    def prepare_prompt_from_ghidra_decompile(self):
        ghidra_decompile = ghdria_decompile_record((self.idx, self.record))
        ghidra_c_code = ghidra_decompile.ghidra_result.ghidra_decompiled_c_code
        asm_code = preprocessing_assembly(self.record["asm"]["code"][-1],
                                          remove_comments=self.remove_comments)
        prompt = GHIDRA_DECOMPILE_TEMPLATE.format(asm_code=asm_code,
                                                  ghidra_c_code=ghidra_c_code)
        self.initial_prompt = prompt
        return prompt

    def get_initial_prompt(self):
        if self.prompt_type == PromptType.SIMILAR_RECORD:
            return self.prepare_prompt_from_similar_record()
        elif self.prompt_type == PromptType.GHIDRA_DECOMPILE:
            return self.prepare_prompt_from_ghidra_decompile()
        else:
            return self.prepare_prompt()

    def decompile_and_evaluate(self, prompt,
                               retry_count: int) -> ResponseValidation:
        """Call the openai api to inference one sample and evaluate the sample

        """
        response_file_path = os.path.join(
            self.output_dir, f"response_{self.idx}.pkl" if retry_count == -1
            else f"response_{self.idx}_retry_{retry_count}.pkl")

        if os.path.exists(response_file_path):
            response = pickle.load(open(response_file_path, "rb"))
        else:
            response = self.llm_client.chat.completions.create(model=self.model_name,
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": prompt
                                                        },
                                                    ],
                                                    n=self.num_generate,
                                                    stream=False,
                                                    timeout=7200)
            # Make sure first save result to persistant storage
            pickle.dump(response, open(response_file_path, "wb"))
        try:
            response_validation = self.evaluate_response(
                prompt, response, retry_count)
            self.retry_response_validation[retry_count] = response_validation
            return response_validation
        except Exception as e:
            logging.warning(
                f"Error in evaluating response for index {self.idx}: {e}")
        return None


    def evaluate_response(self, prompt: str, response: ChatCompletion,
                          retry_count: int) -> ResponseValidation:
        """
        Evaluate the response from the LLM and validate the LLVM IR prediction.
        Returns:
            dict: A dictionary containing the validation results.
        """
        predict_list = extract_llvm_code_from_response(response)
        sample_dir = os.path.join(
            self.output_dir, f"sample_{self.idx}"
            if retry_count == -1 else f"sample_{self.idx}_retry_{retry_count}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)

        def eval_predict(idx, predict):
            predict_execution_success = False
            predict_compile_success, predict_assembly_path, error_msg = compile_llvm_ir(
                predict, os.path.join(sample_dir, f"{idx}"), name_hint="predict")
            assembly = ""
            if predict_compile_success:
                with open(predict_assembly_path, 'r') as f:
                    assembly = f.read()
                    predict_execution_success = eval_assembly(
                        self.record, assembly)
            return EvaluationResult(predict_compile_success,
                                    predict_execution_success, error_msg, predict, assembly)

        # Evaluate the predictions
        for predict in predict_list:
            if isinstance(predict, list) and len(predict) == 0:
                logging.error(f"Empty predict in predict_list: {predict_list}")
        
        with ThreadPoolExecutor(max_workers=len(predict_list)) as executor:
            predict_evaluation_results_list = list[EvaluationResult](
                executor.map(eval_predict, range(len(predict_list)), predict_list))
        # Evaluate the target LLVM ir to make sure the target is correct
        target_compile_success, target_assembly_path, target_error_msg = compile_llvm_ir(
            self.record["llvm_ir"]["code"][-1], sample_dir, name_hint="target")
        target_assembly = ""
        target_execution_success = False
        if target_compile_success:
            with open(target_assembly_path, 'r') as f:
                target_assembly = f.read()
                target_execution_success = eval_assembly(self.record, target_assembly)

        # Summarize the evaluation results
        target_evaluation_result = EvaluationResult(target_compile_success,
                                                    target_execution_success,
                                                    target_error_msg, self.record["llvm_ir"]["code"][-1], target_assembly)
        response_validation = ResponseValidation(
            prompt, response, retry_count, predict_evaluation_results_list,
            target_evaluation_result)
        # Save the response validation to the retry_response_validation dictionary
        self.retry_response_validation[retry_count] = response_validation
        return response_validation


    def get_most_similar_predict(self, retry_count: int) -> EvaluationResult:
        predict_evaluation_results_list = self.retry_response_validation[retry_count - 1].predict_evaluation_results_list
        target_assembly = self.record["asm"]["code"][-1]
        compilable_assembly_list = [(idx, r.assembly) for idx, r in enumerate(predict_evaluation_results_list) if r.compile_success]
        if len(compilable_assembly_list) == 0:
            return predict_evaluation_results_list[0]
        elif len(compilable_assembly_list) == 1:
            return predict_evaluation_results_list[compilable_assembly_list[0][0]]
        query_vector = self.embedding_model.embed([target_assembly])[0]['outputs']['embedding']
        compilable_assembly_vectors = [ v['outputs']['embedding'] for v in self.embedding_model.embed([r[1] for r in compilable_assembly_list]) ]
        query_tensor = torch.tensor(query_vector, dtype=torch.float32)
        compilable_tensors = torch.tensor(np.array(compilable_assembly_vectors), dtype=torch.float32)

        # Calculate cosine similarity between the query vector and all compilable assembly vectors
        # Unsqueeze query_tensor to (1, embedding_dim) to enable batch comparison with F.cosine_similarity
        similarities = torch.nn.functional.cosine_similarity(query_tensor.unsqueeze(0), compilable_tensors)

        # Find the index of the maximum similarity
        most_similar_idx = torch.argmax(similarities).item()

        # Retrieve the assembly string corresponding to the most similar embedding
        most_similar_idx = compilable_assembly_list[most_similar_idx][0]
        return predict_evaluation_results_list[most_similar_idx]


    def prepare_fix_prompt(self, retry_count):
        sample_dir = os.path.join(self.output_dir,
                                  f"sample_{self.idx}_retry_{retry_count}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)
        predict_compile_success_list = [
            r.compile_success for r in self.retry_response_validation[
                retry_count - 1].predict_evaluation_results_list
        ]
        target_asm_code = self.record["asm"]["code"][-1]
        target_asm_code = preprocessing_assembly(
            target_asm_code, remove_comments=self.remove_comments)
        error_msg, fix_idx = None, 0
        predict_list = extract_llvm_code_from_response(
            self.retry_response_validation[retry_count - 1].response)
        # 1. If there is no choice that is compile success
        if not any(predict_compile_success_list):
            error_msg_list = [
                r.error_msg for r in self.retry_response_validation[
                    retry_count - 1].predict_evaluation_results_list
            ]
            predict = ""
            # TODO(Chunwei) Maybe we should use the best prediction instead of the first one
            for choice_idx, error_msg in enumerate(error_msg_list):
                if error_msg is not None and error_msg.strip() != "":
                    error_msg = error_msg.strip()
                    fix_idx = choice_idx
                    predict = predict_list[choice_idx]
                    if isinstance(predict, list) and len(predict) > 0:
                        predict = predict[0]
                    break
            if predict == "":
                predict = predict_list[0]
            prompt = format_compile_error_prompt(self.initial_prompt,
                                                 predict,
                                                 error_msg)
        # 2 If there is one choice that is compile success, we choose the one that is compile success
        # TODO(Chunwei) Maybe we should use the best prediction instead of the first one that is compile success
        else:
            if self.retry_response_validation[retry_count - 1].get_num_compile_success() == 1:
                most_similar_evaluation_result = self.retry_response_validation[retry_count - 1].get_first_compile_success_evaluation_result()
            else:
                most_similar_evaluation_result = self.get_most_similar_predict(retry_count)
            predict = most_similar_evaluation_result.llvm_ir
            predict_assembly = most_similar_evaluation_result.assembly
            predict_assembly = preprocessing_assembly(
                predict_assembly, remove_comments=self.remove_comments)
            prompt = format_execution_error_prompt(self.initial_prompt,
                                                predict,
                                                predict_assembly)
        return prompt

    def correct_one(self) -> bool:
        # TODO(Chunwei) Maybe we should use the best prediction instead of the first one
        retry_count = 0
        fix_success = False
        sample_dir = os.path.join(self.output_dir, f"sample_{self.idx}")
        while retry_count < self.num_retry and not fix_success:
            
            if self.retry_response_validation[retry_count - 1].get_num_execution_success() > 0:
                self.retry_response_validation[retry_count - 1].dump_correct_llvm_ir(sample_dir)
                logger.info(f"Correct LLVM IR saved to {os.path.join(sample_dir, f'correct_llvm_ir.ll')} after {retry_count} retries")
                fix_success = True
                break

            # 2. Prepare the prompt for the next retry
            prompt = self.prepare_fix_prompt(retry_count)

            # 3. Decompile and evaluate the prompt
            self.decompile_and_evaluate(
                prompt, retry_count)
            retry_count += 1
        if not fix_success:
            logger.warning(
                f"Failed to correct LLVM IR for index {self.idx} after {retry_count} retries."
            )
        return fix_success


    def predict_has_compile_and_execution_sucess(self) -> tuple[bool, bool]:
        compile_success_count = False   
        execution_success_count = False
        for response_validation in self.retry_response_validation.values():
            for predict_evaluation_result in response_validation.predict_evaluation_results_list:
                if predict_evaluation_result.compile_success:
                    compile_success_count = True
                if predict_evaluation_result.execution_success:
                    execution_success_count = True
        return compile_success_count, execution_success_count

    def target_has_compile_and_execution_sucess(self) -> tuple[bool, bool]:
        target_compile_success = self.retry_response_validation[-1].target_evaluation_result.compile_success
        target_execution_success = self.retry_response_validation[-1].target_evaluation_result.execution_success
        return target_compile_success, target_execution_success

    def evaluate_existing_output(self):
        """Evaluate the existing output and save the evaluation results to the retry_response_validation dictionary"""
        response_file = os.path.join(self.output_dir, f"response_{self.idx}.pkl")
        sample_dir = os.path.join(self.output_dir, f"sample_{self.idx}")
        if not os.path.exists(sample_dir):
            logger.error(f"Sample directory {sample_dir} does not exist.")
        # 1. Load and evaluate the first response
        response = pickle.load(open(response_file, "rb"))
        self.evaluate_response(self.initial_prompt, response, -1)
        
        # 2. Load and evaluate the retry responses
        for retry_count in range(0, self.num_retry):
            response_file = os.path.join(self.output_dir, f"response_{self.idx}_retry_{retry_count}.pkl")
            if not os.path.exists(response_file):
                continue
            response = pickle.load(open(response_file, "rb"))
            self.evaluate_response(self.initial_prompt, response, retry_count)
            
    