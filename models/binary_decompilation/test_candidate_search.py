"""Multi-candidate selection, context accounting, and native execution regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.binary_decompilation.binary_project import BinaryCoreutilsAdapter, load_config, run_stage
from models.binary_decompilation.project_pipeline import code_block, generate, generation_config, infer_stage, run, write_json
from models.binary_decompilation.prompt_budget import fit_prompt
from models.binary_decompilation.test_native_validation import CLANG


FUNCTION = {"name": "leaf", "function_id": "fixture:0x10", "assembly": "ret",
            "signature": "int leaf(int)", "calls": [], "source_file": "src/fixture.c"}
SETTINGS = {"model": "fixture", "base_url": "http://fixture/v1", "num_generate": 8,
            "temperature": 0.6, "enable_thinking": True, "max_tokens": 1000,
            "context_window": 260000, "max_attempts": 2}


def completion(text, index=0, finish="stop"):
    return {"index": index, "message": {"content": text}, "finish_reason": finish}


def stream_chunk(index, content="", finish=None, reasoning=""):
    item = Mock()
    item.model_dump.return_value = {"choices": [{"index": index,
        "delta": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}]}
    return item


class CandidateSearchTests(unittest.TestCase):
    def test_stream_timeout_preserves_completed_sibling(self):
        import httpx
        client, stream = Mock(), Mock()

        def chunks():
            yield stream_chunk(0, 'int leaf(void) {return 1;}', 'stop')
            yield stream_chunk(1, 'partial')
            raise httpx.ReadTimeout('fixture timeout')

        stream.__iter__ = Mock(side_effect=chunks)
        client.chat.completions.create.return_value = stream
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            response = generate(client, {**SETTINGS, 'num_generate': 2, 'stream': True}, 'prompt', root)
            self.assertTrue((root / 'request_error.json').exists())
        self.assertEqual(response['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(response['choices'][1]['finish_reason'], 'interrupted')
        self.assertEqual(response['request_error']['type'], 'ReadTimeout')

    def test_transport_failure_consumes_round_and_continues(self):
        client, adapter = Mock(), Mock(binary_only=True)
        adapter.validate.return_value = {'accepted': True, 'feedback': 'executed'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('models.binary_decompilation.project_pipeline.generate', side_effect=[
                    TimeoutError('fixture timeout'), {'choices': [completion('int leaf(int x) {return x;}')]}]):
                result = infer_stage(client, SETTINGS, adapter, FUNCTION, {}, 'c', root, 'int leaf(int);')
            self.assertTrue(result['accepted'])
            self.assertEqual(len(list(root.glob('attempt_*/request.json'))), 2)
            self.assertTrue((root / 'attempt_0/request_error.json').exists())

    def test_guided_prompt_contains_compiler_procedure(self):
        client, adapter = Mock(), Mock(binary_only=True)
        adapter.validate.return_value = {'accepted': True, 'feedback': 'executed'}
        client.chat.completions.create.return_value.model_dump.return_value = {'choices': [completion('define void @leaf() {ret void}') ]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_stage(client, {**SETTINGS, 'guided_decompilation': True}, adapter, FUNCTION,
                        {'compiler_cfg': {'blocks': []}}, 'ir', root, 'void leaf(void);')
            prompt = (root / 'attempt_0/prompt.txt').read_text()
        self.assertIn('phi has one incoming value', prompt)
        self.assertIn('compiler_cfg', prompt)
        self.assertIn('external global i32 has NO initializer', prompt)

    def test_prefilled_thinking_does_not_select_a_reasoning_code_example(self):
        text = 'Reasoning with an example:\n```llvm\nwrong\n```\n</think>\n```llvm\ncorrect\n```'
        self.assertEqual(code_block(text, ("llvm",)), "correct\n")
        self.assertEqual(code_block('reasoning\n</think>\n{"answer": 1}', ("json",)), '{"answer": 1}\n')

    def test_c_prompt_uses_accepted_ir_and_contracts_without_redundant_assembly(self):
        client, adapter = Mock(), Mock(binary_only=True)
        client.chat.completions.create.return_value.model_dump.return_value = {
            "choices": [completion("int leaf(int x) { return x; }")]}
        adapter.validate.return_value = {"accepted": True, "feedback": "ok"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_stage(client, {**SETTINGS, "ir_only_c_evidence": True}, adapter, FUNCTION,
                        {"project_header": "int leaf(int);", "runtime_samples": ["unneeded_snapshot"]},
                        "c", root, "int leaf(int);", "accepted_ir_body")
            prompt = (root / "attempt_0/prompt.txt").read_text()
            self.assertIn("accepted_ir_body", prompt)
            self.assertNotIn("unneeded_snapshot", prompt)
            self.assertNotIn('"assembly":', prompt)

    def test_interleaved_stream_choices_and_reasoning_remain_separate(self):
        client, stream = Mock(), Mock()
        stream.__iter__ = Mock(return_value=iter([
            stream_chunk(1, "B", reasoning="think B"), stream_chunk(0, "A", reasoning="think A"),
            stream_chunk(1, "2", "stop"), stream_chunk(0, "1", "stop")]))
        client.chat.completions.create.return_value = stream
        with tempfile.TemporaryDirectory() as tmp:
            response = generate(client, {**SETTINGS, "num_generate": 2, "stream": True}, "prompt", Path(tmp))
        self.assertEqual([c["message"]["content"] for c in response["choices"]], ["A1", "B2"])
        self.assertEqual(response["choices"][1]["message"]["reasoning_content"], "think B")
        options = client.chat.completions.create.call_args.kwargs
        self.assertEqual(options["n"], 2)
        self.assertEqual(options["temperature"], 0.6)
        self.assertTrue(options["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_repetition_in_one_choice_does_not_cancel_other_choice(self):
        client, stream = Mock(), Mock()
        chunks = [stream_chunk(0, f'@.str.{i} = private constant [1 x i8] c"\\00"\n') for i in range(81)]
        chunks += [stream_chunk(1, "valid sibling", "stop")]
        stream.__iter__ = Mock(return_value=iter(chunks))
        client.chat.completions.create.return_value = stream
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(client, {**SETTINGS, "num_generate": 2, "stream": True}, "prompt", Path(tmp))
        self.assertEqual(result["choices"][0]["finish_reason"], "degenerate_constant_repetition")
        self.assertEqual(result["choices"][1]["message"]["content"], "valid sibling")
        self.assertEqual(result["choices"][1]["finish_reason"], "stop")
        stream.close.assert_called_once()

    def test_failed_choice_does_not_hide_later_accepted_choice_or_force_contract_repair(self):
        client, adapter = Mock(), Mock(binary_only=True)
        client.chat.completions.create.return_value.model_dump.return_value = {"choices": [
            completion("```c\nint leaf(int x) { return x; }\n```", 0),
            completion("```c\nint leaf(int x) { return x + 1; }\n```", 1)]}
        adapter.validate.side_effect = [{"accepted": False, "feedback": "project.h error", "repair_contract": True},
                                        {"accepted": True, "feedback": "executed"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = infer_stage(client, SETTINGS, adapter, FUNCTION, {}, "c", root, "int leaf(int);")
            self.assertTrue(result["accepted"])
            self.assertEqual(result["selected_choice_position"], 1)
            self.assertFalse(result.get("repair_contract"))
            self.assertEqual(json.loads((root / "attempt_0/request.json").read_text())["n"], 8)
            self.assertTrue((root / "attempt_0/choice_0/candidate.c").exists())

    def test_next_round_repairs_best_runtime_candidate_not_first_syntax_failure(self):
        client, adapter = Mock(), Mock(binary_only=True)
        first = {"choices": [completion("broken IR", 0), completion("runtime progress", 1)]}
        second = {"choices": [completion("repaired IR", 0)]}
        client.chat.completions.create.return_value.model_dump.side_effect = [first, second]
        adapter.validate.side_effect = [
            {"accepted": False, "feedback": "syntax error"},
            {"accepted": False, "feedback": "wrong output", "execution_observed": True,
             "tests": {"passed": 7, "failed": 1}},
            {"accepted": True, "feedback": "all cases passed"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = infer_stage(client, SETTINGS, adapter, FUNCTION, {}, "ir", root, "int leaf(int);")
            self.assertTrue(result["accepted"])
            prompt = (root / "attempt_1/prompt.txt").read_text()
            self.assertIn("Previous candidate:\nruntime progress", prompt)

    def test_interrupted_batch_consumes_one_round_and_eight_candidate_slots(self):
        client, adapter = Mock(), Mock(binary_only=True, header="", program="fixture",
                                      feedback_config={"programs": {"fixture": {"cases": []}}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(9):
                write_json(root / "old" / f"attempt_{i}/request.json", {"n": 8})
            def fake_infer(client, config, adapter, f, context, stage, directory, signature, ir):
                self.assertEqual(config["max_attempts"], 1)
                write_json(directory / "attempt_0/request.json", {"n": 8})
                return {"accepted": False}
            with patch("models.binary_decompilation.binary_project.infer_stage", side_effect=fake_infer):
                result = run_stage(client, {**SETTINGS, "max_attempts": 10}, adapter, FUNCTION, {}, "ir", root)
            self.assertEqual(result["total_generation_requests"], 10)
            self.assertEqual(result["total_requested_candidates"], 80)
            result = run_stage(client, {**SETTINGS, "max_attempts": 10}, adapter, FUNCTION, {"changed": True}, "ir", root)
            self.assertEqual(result["reason"], "stage generation budget exhausted")

    def test_context_uses_thinking_template_and_per_choice_completion_budget(self):
        tok = Mock()
        tok.apply_chat_template.return_value = [0] * 100
        with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=tok):
            _, budget = fit_prompt({**SETTINGS, "tokenizer_path": "fixture", "max_tokens": 32768}, "IR", {})
        self.assertTrue(tok.apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(budget["input_token_budget"], 260000 - 32768 - 64)

    def test_remaining_context_preserves_evidence_and_caps_completion(self):
        tok = Mock()
        tok.apply_chat_template.return_value = [0] * 17409
        settings = {**SETTINGS, "tokenizer_path": "fixture", "max_tokens": 260000,
                    "max_tokens_policy": "remaining_context"}
        with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=tok):
            prompt, budget = fit_prompt(settings, "IR", {"assembly": "complete_target"})
        self.assertIn("complete_target", prompt)
        self.assertEqual(budget["requested_max_output_tokens"], 260000)
        self.assertEqual(budget["max_output_tokens"], 242527)
        self.assertEqual(budget["prompt_tokens"] + budget["max_output_tokens"] + 64, 260000)

    def test_remaining_context_requires_tokenizer_and_rejects_overflow(self):
        settings = {**SETTINGS, "max_tokens": 260000, "max_tokens_policy": "remaining_context"}
        with self.assertRaisesRegex(ValueError, "requires a local tokenizer"):
            fit_prompt(settings, "IR", {})
        tok = Mock()
        tok.apply_chat_template.return_value = [0] * 260000
        with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=tok):
            with self.assertRaisesRegex(ValueError, "evidence was not truncated"):
                fit_prompt({**settings, "tokenizer_path": "fixture"}, "IR", {})

    def test_effective_output_budget_reaches_single_and_multi_choice_requests(self):
        for count in (1, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                client, adapter, tok = Mock(), Mock(binary_only=True), Mock()
                tok.apply_chat_template.return_value = [0] * 17409
                client.chat.completions.create.return_value.model_dump.return_value = {
                    "choices": [completion("define void @leaf() { ret void }", i) for i in range(count)]}
                adapter.validate.return_value = {"accepted": True, "feedback": "ok"}
                settings = {**SETTINGS, "num_generate": count, "tokenizer_path": "fixture",
                            "max_tokens": 260000, "max_tokens_policy": "remaining_context"}
                with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=tok):
                    result = infer_stage(client, settings, adapter, FUNCTION, {}, "ir", Path(tmp))
                self.assertTrue(result["accepted"], result)
                self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 242527)
                saved = json.loads((Path(tmp) / "attempt_0/generation_config.json").read_text())
                self.assertEqual(saved["max_tokens"], 242527)
                self.assertEqual(saved["requested_max_tokens"], 260000)

    def test_native_batch_accepts_eighth_ir_candidate_and_reuses_it_on_resume(self):
        client = Mock()
        choices = [completion(f'define i32 @leaf(i32 %x) {{ %y = add i32 %x, {i}\nret i32 %y\n}}', i)
                   for i in range(8)]
        client.chat.completions.create.return_value.model_dump.return_value = {"choices": choices}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            source = root / "src/fixture.c"
            source.write_text('#include <stdio.h>\n#include <stdlib.h>\n'
                              '__attribute__((noinline)) int leaf(int x) { return x+7; }\n'
                              'int main(int argc, char **argv) { printf("%d\\n",leaf(atoi(argv[1]))); }\n')
            config = {**SETTINGS, "source_dir": tmp, "build_dir": tmp, "clang": CLANG,
                      "programs": {"fixture": {"tests": [], "cases": [{"args": ["3"]}, {"args": ["-2"]}]}}}
            built = run([CLANG, "-O1", "-fPIE", "-pie", str(source), "-o", str(root / "src/fixture")])
            self.assertEqual(built["returncode"], 0, built)
            adapter = BinaryCoreutilsAdapter(config, "fixture", root)
            adapter.header = "int leaf(int);"
            result = infer_stage(client, config, adapter, FUNCTION, {}, "ir", root / "ir", "int leaf(int);")
            self.assertTrue(result["accepted"], result)
            self.assertEqual(result["selected_choice_position"], 7)
            self.assertTrue(result["validation"]["execution_observed"])
            resumed = infer_stage(client, config, adapter, FUNCTION, {}, "ir", root / "ir", "int leaf(int);")
            self.assertTrue(resumed["accepted"])
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_bad_probe_does_not_reject_valid_signature_and_wrong_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"source_dir": tmp, "build_dir": tmp, "clang": CLANG,
                      "programs": {"fixture": {"tests": [], "cases": []}}}
            adapter = BinaryCoreutilsAdapter(config, "fixture", root)
            probe = {"return_type": "void", "argument_types": ["pointer"],
                     "arguments": [{"buffer": {"byte_count": 16, "id": "buf"}}]}
            write_json(root / "contract.json", {"project_header": "int leaf(int);", "direct_probes": [probe]})
            result = adapter.validate("leaf", "fixture.c", "int leaf(int);", root / "unused.c", root / "check", "signature")
            self.assertTrue(result["accepted"], result)
            self.assertIn("not a nested buffer object", result["probe_errors"][0]["error"])
            self.assertEqual(adapter.direct_probes["leaf"], [])
            write_json(root / "contract.json", {"project_header": "int wrong(int);"})
            result = adapter.validate("leaf", "fixture.c", "int wrong(int);", root / "unused.c", root / "wrong", "signature")
            self.assertFalse(result["accepted"])

    def test_probe_repair_selects_valid_schema_after_malformed_siblings(self):
        client = Mock()
        valid_probe = {"return_type": "void", "argument_types": ["pointer"],
                       "arguments": [{"buffer": 16, "id": "buf", "bytes": [0]}]}
        client.chat.completions.create.return_value.model_dump.return_value = {"choices": [
            completion("[]", 0), completion(json.dumps({"direct_probes": [{"return_type": "integer"}]}), 1),
            completion(json.dumps({"direct_probes": [valid_probe]}), 2)]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {**SETTINGS, "source_dir": tmp, "build_dir": tmp, "clang": CLANG,
                      "programs": {"fixture": {"tests": [], "cases": []}}}
            adapter = BinaryCoreutilsAdapter(config, "fixture", root)
            result = infer_stage(client, config, adapter, FUNCTION, {}, "probes", root / "probes", "void leaf(char *);")
            self.assertTrue(result["accepted"], result)
            self.assertEqual(result["selected_choice_position"], 2)
            self.assertEqual(adapter.direct_probes["leaf"], [valid_probe])

    def test_new_preset_is_separate_and_matches_server_window(self):
        config = load_config(Path("models/binary_decompilation/configs/coreutils_binary_candidates8.json"))
        self.assertEqual(config["context_window"], 260000)
        self.assertEqual(config["num_generate"], 8)
        self.assertEqual(config["max_tokens_policy"], "remaining_context")
        for stage in ("signature", "probes", "ir", "c", "integration"):
            self.assertEqual(generation_config(config, stage)["max_tokens"], 260000)
        self.assertTrue(config["enable_thinking"])
        self.assertFalse(generation_config(config, "signature")["enable_thinking"])
        self.assertFalse(generation_config(config, "probes")["enable_thinking"])
        self.assertTrue(generation_config(config, "ir")["enable_thinking"])
        self.assertTrue(generation_config(config, "c")["enable_thinking"])
        self.assertNotEqual(config["output_dir"], config["evidence_cache"])


if __name__ == "__main__":
    unittest.main()
