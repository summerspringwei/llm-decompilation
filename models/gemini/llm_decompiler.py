"""Core LLM-based decompilation pipeline.

Classes
-------
``EvaluationResult``
    Outcome of compiling and executing a single LLVM IR prediction.

``ResponseValidation``
    Container linking an LLM response to its evaluated predictions.

``LLMDecompileRecord``
    Orchestrates the decompile → evaluate → fix loop for a single sample.

This module was refactored from the original monolithic version.  Prompt
construction is now delegated to :mod:`utils.prompt_builder`, response
parsing to :mod:`utils.llm_response_parser`, and subprocess calls to
:mod:`utils.subprocess_utils`.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import faulthandler
import signal

import numpy as np
import requests
import torch

from openai.types.chat.chat_completion import ChatCompletion

from config import DecompilationConfig
from models.ghidra_decompile.ghidra_decompile_exebench import (
    ghidra_decompile_record,
)
from models.rag.exebench_qdrant_base import ExebenchQdrantSearch
from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.exebench_sample import ExebenchSample
from utils.llm_response_parser import (
    extract_llvm_code_from_response,
    strip_think_block,
)
from utils.logging_config import get_logger
from utils.preprocessing_assembly import preprocessing_assembly
from utils.preprocessing_llvm_ir import preprocessing_llvm_ir
from utils.prompt_builder import (
    build_basic_prompt,
    build_compile_error_prompt,
    build_execution_error_prompt,
    build_execution_error_prompt_with_ghidra_decompile,
    build_failure_analysis_prompt,
    build_ghidra_decompile_prompt,
    build_llm_fix_prompt,
    build_pcode_prompt,
    build_pcode_similar_record_prompt,
    build_similar_record_prompt,
)
from utils.prompt_type import PromptType
from utils.subprocess_utils import run_command

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class EvaluationResult:
    """Outcome of compiling and optionally executing a single LLVM IR prediction."""

    def __init__(
        self,
        compile_success: bool,
        execution_success: bool,
        error_msg: str,
        llvm_ir: Optional[str] = None,
        assembly: Optional[str] = None,
    ):
        self.compile_success = compile_success
        self.execution_success = execution_success
        self.error_msg = error_msg
        self.llvm_ir = llvm_ir
        self.assembly = assembly


class ResponseValidation:
    """Links an LLM response to the evaluation outcomes for each choice."""

    def __init__(
        self,
        prompt: str,
        response: ChatCompletion,
        retry_count: int,
        predict_evaluation_results_list: list[EvaluationResult],
        target_evaluation_result: EvaluationResult,
    ):
        self.prompt = prompt
        self.response = response
        self.retry_count = retry_count
        self.predict_evaluation_results_list = predict_evaluation_results_list
        self.target_evaluation_result = target_evaluation_result

    # -- Aggregate queries ---------------------------------------------------

    def get_num_execution_success(self) -> int:
        return sum(
            1
            for r in self.predict_evaluation_results_list
            if r.execution_success
        )

    def get_num_compile_success(self) -> int:
        return sum(
            1
            for r in self.predict_evaluation_results_list
            if r.compile_success
        )

    def get_first_compile_success_evaluation_result(
        self,
    ) -> Optional[EvaluationResult]:
        return next(
            (r for r in self.predict_evaluation_results_list if r.compile_success),
            None,
        )

    def get_num_total(self) -> int:
        return len(self.predict_evaluation_results_list)

    def dump_correct_llvm_ir(self, output_dir: str) -> None:
        for result in self.predict_evaluation_results_list:
            if result.execution_success:
                path = os.path.join(output_dir, "correct_llvm_ir.ll")
                with open(path, "w") as f:
                    f.write(result.llvm_ir)
                break


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class LLMDecompileRecord:
    """Orchestrates the decompile → evaluate → fix loop for one sample.

    Args:
        record: An ExeBench sample (dict or :class:`ExebenchSample`).
        idx: Numeric index of this sample in the dataset.
        config: Runtime configuration.
        llm_client: An ``openai.OpenAI`` client instance.
        model_name: Model identifier to pass to the API.
        rag_search: Pre-initialised RAG search helper (may be ``None``).
    """

    def __init__(
        self,
        record,
        idx: int,
        config: DecompilationConfig,
        llm_client,
        model_name: str,
        rag_search: Optional[ExebenchQdrantSearch] = None,
    ):
        if isinstance(record, ExebenchSample):
            self.record = record
        else:
            self.record = ExebenchSample.from_dict(record)

        self.idx = idx
        self.config = config
        self.llm_client = llm_client
        self.model_name = model_name
        self.prompt_type = PromptType(config.prompt_type)
        self.fix_prompt_type = PromptType(config.fix_prompt_type)

        self.retry_response_validation: dict[int, ResponseValidation] = {}
        self.similar_record: Optional[ExebenchSample] = None
        self.pcode_text: Optional[str] = None
        self.similar_pcode_text: Optional[str] = None
        self.score: Optional[float] = None
        self.initial_prompt: Optional[str] = None

        self.exebench_qdrant_search = rag_search

    # -- LLM call helpers ----------------------------------------------------

    def _save_and_load_response(
        self,
        response_path: str,
        prompt: str,
    ) -> ChatCompletion:
        """Load a cached response or call the LLM and persist the result."""
        if os.path.exists(response_path):
            with open(response_path, "rb") as f:
                return pickle.load(f)

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            n=self.config.num_generate,
            stream=False,
            timeout=self.config.llm_timeout,
        )
        # Persist *before* any post-processing in case of crash.
        with open(response_path, "wb") as f:
            pickle.dump(response, f)
        return response

    # -- Ghidra P-code -------------------------------------------------------

    def _generate_pcode_text(self, record: ExebenchSample, label: str) -> str:
        """Generate Ghidra P-code text from assembly via a temp-dir pipeline."""
        asm_code = record.asm.code[-1]
        func_name = record.fname
        ghidra_script = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "ghidra_decompile",
                "ghidra_pcode_script.py",
            )
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            asm_path = os.path.join(tmp_dir, f"{func_name}.s")
            obj_path = os.path.join(tmp_dir, f"{func_name}.o")
            out_path = os.path.join(tmp_dir, "pcode.txt")
            with open(asm_path, "w") as f:
                f.write(asm_code)

            retcode, _, stderr = run_command(
                ["clang", "-c", asm_path, "-o", obj_path], timeout=30
            )
            if retcode != 0:
                raise RuntimeError(
                    f"Failed to compile assembly to object file: {stderr}"
                )

            project_dir = os.path.join(
                tmp_dir, f"ghidra_project_{self.idx}_{label}"
            )
            os.makedirs(project_dir, exist_ok=True)
            project_name = f"project_{self.idx}_{label}"
            cmd = [
                self.config.ghidra.headless_analyzer_path,
                project_dir,
                project_name,
                "-import",
                obj_path,
                "-overwrite",
                "-postscript",
                ghidra_script,
                func_name,
                out_path,
            ]
            retcode, _, stderr = run_command(
                cmd, timeout=self.config.ghidra.command_timeout
            )
            if retcode != 0:
                raise RuntimeError(
                    f"Ghidra P-code generation failed: {stderr}"
                )
            with open(out_path, "r") as f:
                return f.read()

    def _get_pcode_text(
        self,
        record: ExebenchSample,
        label: str,
        cache_attr: str,
    ) -> str:
        cached = getattr(self, cache_attr)
        if cached:
            return cached
        pcode = self._generate_pcode_text(record, label)
        setattr(self, cache_attr, pcode)
        return pcode

    # -- Prompt preparation --------------------------------------------------

    def _prepare_basic_prompt(self) -> str:
        if self.config.use_pcode:
            p_code = self._get_pcode_text(
                self.record, f"record_{self.idx}", "pcode_text"
            )
            return build_pcode_prompt(p_code)
        asm_code = preprocessing_assembly(
            self.record.asm.code[-1],
            remove_comments=self.config.remove_comments,
        )
        return build_basic_prompt(asm_code)

    def _prepare_prompt_from_similar_record(self) -> str:
        similar_record, self.score = (
            self.exebench_qdrant_search.find_similar_records_in_exebench_synth_rich_io(
                self.record.to_dict()
            )
        )
        self.similar_record = ExebenchSample.from_dict(similar_record)

        if self.config.use_pcode:
            p_code = self._get_pcode_text(
                self.record, f"record_{self.idx}", "pcode_text"
            )
            similar_pcode = self._get_pcode_text(
                self.similar_record, f"similar_{self.idx}", "similar_pcode_text"
            )
            similar_llvm_ir = preprocessing_llvm_ir(
                self.similar_record.llvm_ir.code[-1]
            )
            prompt = build_pcode_similar_record_prompt(
                p_code, similar_pcode, similar_llvm_ir
            )
        else:
            asm_code = preprocessing_assembly(
                self.record.asm.code[-1],
                remove_comments=self.config.remove_comments,
            )
            similar_asm_code = preprocessing_assembly(
                self.similar_record.asm.code[-1],
                remove_comments=self.config.remove_comments,
            )
            similar_llvm_ir = preprocessing_llvm_ir(
                self.similar_record.llvm_ir.code[-1]
            )
            prompt = build_similar_record_prompt(
                asm_code, similar_asm_code, similar_llvm_ir
            )

        # Persist the similar record for debugging.
        similar_path = os.path.join(
            self.config.output_dir, f"similar_record_{self.idx}.pkl"
        )
        with open(similar_path, "wb") as f:
            pickle.dump(self.similar_record, f)

        return prompt

    def _prepare_prompt_from_ghidra_decompile(self) -> str:
        ghidra_decompiler = ghidra_decompile_record(
            (self.idx, self.record.to_dict())
        )
        ghidra_c_code = (
            ghidra_decompiler.ghidra_result.ghidra_decompiled_c_code
        )
        asm_code = preprocessing_assembly(
            self.record.asm.code[-1],
            remove_comments=self.config.remove_comments,
        )
        return build_ghidra_decompile_prompt(asm_code, ghidra_c_code)

    def get_initial_prompt(self) -> str:
        """Build and cache the initial decompilation prompt.

        The strategy is selected based on ``config.prompt_type`` and
        ``config.use_pcode``.
        """
        if self.config.use_pcode and self.prompt_type == PromptType.GHIDRA_DECOMPILE:
            self.initial_prompt = self._prepare_basic_prompt()
        elif self.prompt_type == PromptType.SIMILAR_RECORD:
            self.initial_prompt = self._prepare_prompt_from_similar_record()
        elif self.prompt_type == PromptType.GHIDRA_DECOMPILE:
            self.initial_prompt = self._prepare_prompt_from_ghidra_decompile()
        else:
            self.initial_prompt = self._prepare_basic_prompt()
        return self.initial_prompt

    # -- Decompile + evaluate ------------------------------------------------

    def decompile_and_evaluate(
        self,
        prompt: str,
        retry_count: int,
    ) -> Optional[ResponseValidation]:
        """Call the LLM, evaluate all choices, and cache results.

        Args:
            prompt: The prompt to send.
            retry_count: ``-1`` for the initial attempt, ``0..N`` for retries.

        Returns:
            A :class:`ResponseValidation` or ``None`` on fatal evaluation error.
        """
        suffix = (
            f"sample_{self.idx}"
            if retry_count == -1
            else f"sample_{self.idx}_retry_{retry_count}"
        )
        sample_dir = os.path.join(self.config.output_dir, suffix)
        os.makedirs(sample_dir, exist_ok=True)

        # Save prompt text.
        prompt_txt_path = os.path.join(sample_dir, "prompt.txt")
        if not os.path.exists(prompt_txt_path):
            with open(prompt_txt_path, "w") as f:
                f.write(prompt)

        # Save prompt pickle.
        prompt_pkl_name = (
            f"prompt_{self.idx}.pkl"
            if retry_count == -1
            else f"prompt_{self.idx}_retry_{retry_count}.pkl"
        )
        prompt_pkl_path = os.path.join(self.config.output_dir, prompt_pkl_name)
        if not os.path.exists(prompt_pkl_path):
            with open(prompt_pkl_path, "wb") as f:
                pickle.dump(prompt, f)

        # Get or call LLM response.
        response_name = (
            f"response_{self.idx}.pkl"
            if retry_count == -1
            else f"response_{self.idx}_retry_{retry_count}.pkl"
        )
        response_path = os.path.join(self.config.output_dir, response_name)
        response = self._save_and_load_response(response_path, prompt)

        try:
            validation = self.evaluate_response(prompt, response, retry_count)
            self.retry_response_validation[retry_count] = validation
            return validation
        except Exception as e:
            logger.warning(
                "Error evaluating response for index %d: %s", self.idx, e
            )
        return None

    def evaluate_response(
        self,
        prompt: str,
        response: ChatCompletion,
        retry_count: int,
    ) -> ResponseValidation:
        """Evaluate all predictions in *response* against the ground truth."""
        predict_list = extract_llvm_code_from_response(response)

        suffix = (
            f"sample_{self.idx}"
            if retry_count == -1
            else f"sample_{self.idx}_retry_{retry_count}"
        )
        sample_dir = os.path.join(self.config.output_dir, suffix)
        os.makedirs(sample_dir, exist_ok=True)

        # Sanity-check.
        for predict in predict_list:
            if isinstance(predict, list) and len(predict) == 0:
                logger.error("Empty predict in predict_list: %s", predict_list)

        def _eval_single_predict(idx, predict):
            faulthandler.register(signal.SIGUSR1)
            execution_success = False
            compile_success, assembly_path, error_msg = compile_llvm_ir(
                predict, os.path.join(sample_dir, f"{idx}"), name_hint="predict"
            )
            assembly = ""
            if compile_success:
                with open(assembly_path, "r") as f:
                    assembly = f.read()
                    execution_success = eval_assembly(
                        self.record.to_dict(), assembly
                    )
            return EvaluationResult(
                compile_success, execution_success, error_msg, predict, assembly
            )

        with ThreadPoolExecutor(
            max_workers=len(predict_list)
        ) as executor:
            predict_results = list(
                executor.map(
                    _eval_single_predict,
                    range(len(predict_list)),
                    predict_list,
                )
            )

        # Evaluate the target LLVM IR.
        target_success, target_asm_path, target_error = compile_llvm_ir(
            self.record.llvm_ir.code[-1], sample_dir, name_hint="target"
        )
        target_assembly = ""
        target_exec_success = False
        if target_success:
            with open(target_asm_path, "r") as f:
                target_assembly = f.read()
                target_exec_success = eval_assembly(
                    self.record.to_dict(), target_assembly
                )

        target_result = EvaluationResult(
            target_success,
            target_exec_success,
            target_error,
            self.record.llvm_ir.code[-1],
            target_assembly,
        )
        validation = ResponseValidation(
            prompt, response, retry_count, predict_results, target_result
        )
        self.retry_response_validation[retry_count] = validation
        return validation

    # -- Similarity search ---------------------------------------------------

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        """Fetch embeddings from the remote embedding service."""
        response = requests.post(self.config.rag.embedding_url, json=texts)
        response.raise_for_status()
        return response.json()["embeddings"]

    def get_most_similar_predict(
        self, retry_count: int
    ) -> EvaluationResult:
        """Among compilable predictions, return the one most similar to the target assembly."""
        prev = self.retry_response_validation[retry_count - 1]
        target_assembly = self.record.asm.code[-1]
        compilable = [
            (i, r.assembly)
            for i, r in enumerate(prev.predict_evaluation_results_list)
            if r.compile_success
        ]

        if len(compilable) == 0:
            return prev.predict_evaluation_results_list[0]
        if len(compilable) == 1:
            return prev.predict_evaluation_results_list[compilable[0][0]]

        best_idx_in_compilable = 0
        try:
            all_texts = [target_assembly] + [c[1] for c in compilable]
            all_vectors = self._get_embedding(all_texts)
            query = torch.tensor(all_vectors[0], dtype=torch.float32)
            candidates = torch.tensor(
                np.array(all_vectors[1:]), dtype=torch.float32
            )
            similarities = torch.nn.functional.cosine_similarity(
                query, candidates
            )
            best_idx_in_compilable = torch.argmax(similarities).item()
            logger.info(
                "Most similar index: %d out of %d",
                best_idx_in_compilable,
                len(compilable),
            )
        except Exception as e:
            logger.error("Error in similarity search: %s", e)

        if best_idx_in_compilable >= len(compilable):
            logger.warning(
                "Similarity index %d out of range %d",
                best_idx_in_compilable,
                len(compilable),
            )
            original_idx = compilable[0][0]
        else:
            original_idx = compilable[best_idx_in_compilable][0]

        logger.info(
            "Chose most similar index: %d from %s",
            original_idx,
            [c[0] for c in compilable],
        )
        return prev.predict_evaluation_results_list[original_idx]

    # -- Fix-prompt preparation ----------------------------------------------

    def prepare_compile_fix_prompt(self, retry_count: int) -> str:
        """Build a fix prompt for the given retry round."""
        sample_dir = os.path.join(
            self.config.output_dir, f"sample_{self.idx}_retry_{retry_count}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        prev = self.retry_response_validation[retry_count - 1]
        has_compile_success = any(
            r.compile_success for r in prev.predict_evaluation_results_list
        )
        predict_list = extract_llvm_code_from_response(prev.response)

        if not has_compile_success:
            # Pick the first prediction with a non-empty error message.
            predict = ""
            error_msg = ""
            for choice_idx, r in enumerate(prev.predict_evaluation_results_list):
                if r.error_msg and r.error_msg.strip():
                    error_msg = r.error_msg.strip()
                    predict = predict_list[choice_idx]
                    if isinstance(predict, list) and len(predict) > 0:
                        predict = predict[0]
                    break
            if not predict:
                predict = predict_list[0]
            return build_compile_error_prompt(
                self.initial_prompt, predict, error_msg
            )

        # At least one choice compiled — pick the most similar one.
        if prev.get_num_compile_success() == 1:
            best = prev.get_first_compile_success_evaluation_result()
            logger.info("Only one choice compiled successfully")
        else:
            best = self.get_most_similar_predict(retry_count)
            logger.info("Using most similar compilable prediction")

        predict_llvm_ir = best.llvm_ir
        predict_assembly = preprocessing_assembly(
            best.assembly, remove_comments=self.config.remove_comments
        )

        if self.prompt_type in (
            PromptType.GHIDRA_DECOMPILE,
            PromptType.SIMILAR_RECORD,
        ):
            return build_execution_error_prompt(
                self.initial_prompt, predict_llvm_ir, predict_assembly
            )
        elif self.prompt_type == PromptType.GHIDRA_DECOMPILE_WITH_PREDICT:
            decompiler = ghidra_decompile_record(
                (self.idx, self.record.to_dict())
            )
            ghidra_c_code = decompiler.decompile_external_assembly(
                predict_assembly,
                self.record.func_info.functions[0]["name"],
            )
            return build_execution_error_prompt_with_ghidra_decompile(
                self.initial_prompt, predict_assembly, ghidra_c_code
            )
        else:
            raise ValueError(f"Unsupported prompt type for fix: {self.prompt_type}")

    def _get_best_compilable_result(
        self, retry_count: int
    ) -> Optional[EvaluationResult]:
        """Search all retries for a compilable prediction."""
        prev = self.retry_response_validation.get(retry_count - 1)
        if prev:
            best = prev.get_first_compile_success_evaluation_result()
            if best:
                return best
        # Broaden search to all retries.
        for _, rv in self.retry_response_validation.items():
            for er in rv.predict_evaluation_results_list:
                if er.compile_success:
                    return er
        logger.error(
            "No compilable prediction found for index %d after %d retries.",
            self.idx,
            retry_count,
        )
        return None

    def prepare_llm_fix_prompt(self, retry_count: int) -> Optional[str]:
        """Two-stage LLM fix: analyse failure then ask for correction."""
        best = self._get_best_compilable_result(retry_count)
        if best is None:
            return None

        analysis_prompt = build_failure_analysis_prompt(
            target_assembly=self.record.asm.code[-1],
            predicted_llvm_ir=best.llvm_ir or "",
            predicted_assembly=best.assembly or "",
            compile_error=best.error_msg or "",
        )
        logger.info("Failure analysis prompt built (%d chars)", len(analysis_prompt))

        # Call LLM for the analysis step.
        suffix = (
            f"response_{self.idx}_fix.pkl"
            if retry_count == -1
            else f"response_{self.idx}_retry_{retry_count}_fix.pkl"
        )
        response_path = os.path.join(self.config.output_dir, suffix)
        response = self._save_and_load_response(response_path, analysis_prompt)

        fix_text = (
            response.choices[0].message.content
            if response.choices and response.choices[0].message
            else ""
        )
        fix_text = strip_think_block(fix_text)
        logger.info("Analysis response: %s", fix_text[:200])

        return build_llm_fix_prompt(
            self.initial_prompt,
            best.llvm_ir or "",
            best.assembly or "",
            fix_text,
        )

    # -- Iterative correction loop -------------------------------------------

    def correct_one(self) -> bool:
        """Run the iterative fix loop up to ``config.num_retry`` times."""
        sample_dir = os.path.join(self.config.output_dir, f"sample_{self.idx}")
        os.makedirs(sample_dir, exist_ok=True)

        for retry_count in range(self.config.num_retry):
            prev = self.retry_response_validation.get(retry_count - 1)
            if prev and prev.get_num_execution_success() > 0:
                prev.dump_correct_llvm_ir(sample_dir)
                logger.info(
                    "Correct LLVM IR saved after %d retries to %s",
                    retry_count,
                    sample_dir,
                )
                return True

            # Build next fix prompt.
            if (
                self.fix_prompt_type == PromptType.LLM_FIX
                and self._get_best_compilable_result(retry_count) is not None
            ):
                prompt = self.prepare_llm_fix_prompt(retry_count)
            else:
                prompt = self.prepare_compile_fix_prompt(retry_count)

            if prompt is None:
                logger.warning("Could not build fix prompt for retry %d", retry_count)
                continue

            self.decompile_and_evaluate(prompt, retry_count)

        logger.warning(
            "Failed to correct LLVM IR for index %d after %d retries.",
            self.idx,
            self.config.num_retry,
        )
        return False

    # -- Query helpers -------------------------------------------------------

    def predict_has_compile_and_execution_success(self) -> tuple[bool, bool]:
        """Check across all retries whether any prediction compiled / executed."""
        any_compile = False
        any_execute = False
        for rv in self.retry_response_validation.values():
            for er in rv.predict_evaluation_results_list:
                if er.compile_success:
                    any_compile = True
                if er.execution_success:
                    any_execute = True
        return any_compile, any_execute

    # Backward compat alias (fixes the old typo).
    predict_has_compile_and_execution_sucess = (
        predict_has_compile_and_execution_success
    )

    def target_has_compile_and_execution_success(self) -> tuple[bool, bool]:
        """Check whether the ground-truth LLVM IR compiled and executed."""
        target = self.retry_response_validation[-1].target_evaluation_result
        return target.compile_success, target.execution_success

    target_has_compile_and_execution_sucess = (
        target_has_compile_and_execution_success
    )

    def execution_success(self) -> bool:
        """Return ``True`` if *any* prediction across all retries executed."""
        return any(
            any(r.execution_success for r in rv.predict_evaluation_results_list)
            for rv in self.retry_response_validation.values()
        )

    def compile_success(self) -> bool:
        """Return ``True`` if *any* prediction across all retries compiled."""
        return any(
            any(r.compile_success for r in rv.predict_evaluation_results_list)
            for rv in self.retry_response_validation.values()
        )

    def get_execution_success_retry_count(self) -> Optional[int]:
        """Return the retry index where execution first succeeded, or ``None``."""
        for idx, rv in self.retry_response_validation.items():
            if any(r.execution_success for r in rv.predict_evaluation_results_list):
                return idx
        return None

    def get_execution_success_evaluation_result(
        self,
    ) -> Optional[EvaluationResult]:
        """Return the first prediction that executed successfully."""
        for rv in self.retry_response_validation.values():
            for r in rv.predict_evaluation_results_list:
                if r.execution_success:
                    return r
        return None

    def evaluate_existing_output(self) -> None:
        """Re-evaluate previously saved responses (useful for re-scoring)."""
        response_file = os.path.join(
            self.config.output_dir, f"response_{self.idx}.pkl"
        )
        sample_dir = os.path.join(self.config.output_dir, f"sample_{self.idx}")
        if not os.path.exists(sample_dir):
            logger.error("Sample directory %s does not exist.", sample_dir)

        with open(response_file, "rb") as f:
            response = pickle.load(f)
        self.evaluate_response(self.initial_prompt, response, -1)

        for retry_count in range(self.config.num_retry):
            path = os.path.join(
                self.config.output_dir,
                f"response_{self.idx}_retry_{retry_count}.pkl",
            )
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                response = pickle.load(f)
            self.evaluate_response(self.initial_prompt, response, retry_count)

    # -- Cleanup for multiprocessing -----------------------------------------

    def finalize(self) -> None:
        """Remove unpickleable objects before returning from a worker."""
        self.llm_client = None
        self.exebench_qdrant_search = None