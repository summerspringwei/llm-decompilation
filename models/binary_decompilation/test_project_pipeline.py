"""CPU-side pipeline checks; no inference results are claimed by these tests."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.binary_decompilation.project_pipeline import (
    CoreutilsAdapter, Retriever, code_block, dependency_components, infer_stage, model_client, run, trace_cases, write_json,
)
from models.binary_decompilation.serve_when_free import active_compute_gpus, gpu_memory
from models.binary_decompilation.export_source_references import find_definition_span
from models.binary_decompilation.prompt_budget import fit_prompt
from models.binary_decompilation.service_preflight import check_services


class PipelineTests(unittest.TestCase):
    def test_preflight_rejects_wrong_model_without_generating(self):
        client = Mock()
        client.models.list.return_value.data = []
        config = {"base_url": "http://remote/v1", "model": "wanted", "retrieval": {"enabled": False}}
        with patch("models.binary_decompilation.service_preflight.run", return_value={"returncode": 0}):
            result = check_services(client, config)
        self.assertFalse(result["ready"])
        self.assertEqual(result["inference_generation_requests"], 0)
        client.chat.completions.create.assert_not_called()

    def test_preflight_requires_sufficient_server_context_and_tracing(self):
        client = Mock()
        model = Mock(id="wanted", max_model_len=8192)
        client.models.list.return_value.data = [model]
        config = {"base_url": "http://remote/v1", "model": "wanted", "context_window": 131072,
                  "retrieval": {"enabled": False}}
        with patch("models.binary_decompilation.service_preflight.run", return_value={"returncode": 1}):
            result = check_services(client, config)
        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["inference"]["ok"])
        self.assertFalse(result["checks"]["runtime_tracing"]["ok"])

    def test_remote_model_uses_environment_authentication(self):
        config = {"base_url": "http://10.208.130.107:9001/v1", "api_key_env": "QWEN_PROJECT_API_KEY"}
        with patch.dict("os.environ", {"QWEN_PROJECT_API_KEY": "fixture-secret"}):
            with patch("models.binary_decompilation.project_pipeline.OpenAI") as client:
                model_client(config)
                self.assertEqual(client.call_args.kwargs["api_key"], "fixture-secret")
                self.assertEqual(client.call_args.kwargs["base_url"], config["base_url"])
                self.assertEqual(client.call_args.kwargs["max_retries"], 0)

    def test_missing_remote_auth_fails_before_network_request(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("models.binary_decompilation.project_pipeline.OpenAI") as client:
                with self.assertRaisesRegex(ValueError, "QWEN_PROJECT_API_KEY"):
                    model_client({"base_url": "http://remote/v1", "api_key_env": "QWEN_PROJECT_API_KEY"})
                client.assert_not_called()

    def test_resume_revalidates_without_regeneration_and_guards_model_identity(self):
        client, adapter = Mock(), Mock()
        contract = {"c_declaration": "int leaf(void);", "memory_layouts": [], "uncertainties": []}
        client.chat.completions.create.return_value.model_dump.return_value = {
            "choices": [{"message": {"content": json.dumps(contract)}}]}
        adapter.validate.return_value = {"accepted": True, "feedback": "ok"}
        config = {"model": "fixture", "max_attempts": 2, "max_tokens": 64}
        function = {"name": "leaf", "assembly": "ret", "signature": "int leaf(void)",
                    "calls": [], "source_file": "src/fixture.c"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = infer_stage(client, config, adapter, function, {}, "signature", root)
            original_prompt = (root / "attempt_0/prompt.txt").read_text()
            second = infer_stage(client, config, adapter, function, {}, "signature", root)
            self.assertEqual(first["contract"], second["contract"])
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(adapter.validate.call_count, 2)
            self.assertEqual((root / "attempt_0/prompt.txt").read_text(), original_prompt)
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                infer_stage(client, {**config, "model": "different"}, adapter, function, {}, "signature", root)

    def test_prompt_budget_drops_examples_not_target_evidence(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(messages[0]["content"])
        evidence = {"target": "f", "assembly": "required assembly", "retrieved_examples": [
            {"index": 1, "c": "x" * 1000}]}
        with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=Tokenizer()):
            prompt, budget = fit_prompt({"tokenizer_path": "fixture", "context_window": 512,
                                         "max_tokens": 64}, "Translate", evidence)
        self.assertIn("required assembly", prompt)
        self.assertEqual(len(budget["removed_retrieval_examples"]), 1)
        self.assertEqual(len(evidence["retrieved_examples"]), 1)

    def test_oversize_target_fails_without_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(messages[0]["content"])
        with patch("models.binary_decompilation.prompt_budget.tokenizer", return_value=Tokenizer()):
            with self.assertRaisesRegex(ValueError, "evidence was not truncated"):
                fit_prompt({"tokenizer_path": "fixture", "context_window": 200, "max_tokens": 64},
                           "Translate", {"assembly": "a" * 300})

    def test_runtime_probe_preserves_arguments_and_records_registers(self):
        config = json.loads(Path("models/binary_decompilation/configs/coreutils_qwen38.json").read_text())
        config["programs"]["basename"]["cases"] = [{"args": ["--help", "a b"]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = CoreutilsAdapter(config, "basename", root)
            def traced(command, **kwargs):
                (root / "trace-events.jsonl").write_text(json.dumps(
                    {"stack": ["main", "_start"], "registers": {"rdi": 3}}) + "\n")
                return {"returncode": 0, "stdout": "", "stderr": "", "command": command}
            with patch("models.binary_decompilation.project_pipeline.run", side_effect=traced):
                result = trace_cases(adapter, ["main"], root)
            self.assertTrue(result["trace_valid"])
            self.assertEqual(result["entry_register_samples"]["main"], [{"rdi": 3}])
            command = result["invocations"][0]["command"]
            self.assertTrue(any(arg.startswith("run --help 'a b' < ") for arg in command))

    def test_no_retrieval_does_not_contact_services(self):
        with patch("requests.post", side_effect=AssertionError("Unexpected network request")):
            self.assertEqual(Retriever({"enabled": False}).retrieve(Path("missing"), "f"), [])

    def test_definition_span_preserves_headers_and_preceding_declarations(self):
        source = '#include <stdio.h>\nint global;\nstatic int\nf(int x) { return x; }\n'
        span = find_definition_span(source, "f")
        self.assertEqual(source[span[0]:span[1]], "static int\nf(int x) { return x; }")

    def test_real_ir_and_c_replacement_reject_bad_behavior(self):
        config = json.loads(Path("models/binary_decompilation/configs/coreutils_qwen38.json").read_text())
        class FixtureAdapter(CoreutilsAdapter):
            def compile_source(self, source, output, optimization="-O3"):
                return run([self.clang, optimization, "-c", str(source), "-o", str(output)])

            def link(self, objects, executable, source_file):
                return run([self.clang, *map(str, objects), "-o", str(executable)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src/fixture.c"
            source.write_text('#include <stdlib.h>\n#include <stdio.h>\n'
                              'int transform(int x) { return x + 1; }\n'
                              'int main(int argc, char **argv) { '
                              'printf("%d\\n", transform(atoi(argv[1]))); return 0; }\n')
            config.update(source_dir=str(root), build_dir=str(root),
                          programs={"fixture": {"tests": [], "cases": [
                              {"args": ["6"]}, {"args": ["-3"]}]}})
            result = run([config["clang"], str(source), "-o", str(root / "src/fixture")])
            self.assertEqual(result["returncode"], 0, result)
            adapter = FixtureAdapter(config, "fixture", root)
            ir = root / "candidate.ll"
            ir.write_text('define i32 @transform(i32 %x) {\n'
                          '  %y = add i32 %x, 1\n  ret i32 %y\n}\n')
            result = adapter.validate("transform", "src/fixture.c", "int transform(int x);",
                                      ir, root / "ir-validation", "ir")
            self.assertTrue(result["accepted"], result)
            candidate = root / "candidate.c"
            candidate.write_text("int transform(int x) { return x + 2; }\n")
            result = adapter.validate("transform", "src/fixture.c", "int transform(int x);",
                                      candidate, root / "bad-c-validation", "c")
            self.assertFalse(result["accepted"], result)
            candidate.write_text("int transform(int x) { return x + 1; }\n")
            result = adapter.integrate([{"name": "transform", "source_file": "src/fixture.c",
                                         "candidate": str(candidate)}], root / "integration")
            self.assertTrue(result["accepted"], result)

    def test_components_separate_recursion_from_callers(self):
        functions = [{"name": name, "calls": [{"name": callee} for callee in calls]}
                     for name, calls in [("root", ["a", "leaf"]), ("a", ["b"]),
                                         ("b", ["a"]), ("leaf", [])]]
        components = dependency_components(functions, {"root", "a", "b", "leaf"})
        self.assertIn(["a", "b"], components)
        self.assertEqual(components[-1], ["root"])
        self.assertEqual(len(components), 3)

    def test_empty_graph(self):
        self.assertEqual(dependency_components([], set()), [])

    def test_code_extraction(self):
        self.assertEqual(code_block("```llvm\nret void\n```", ("llvm",)), "ret void\n")
        with self.assertRaises(ValueError):
            code_block("```python\npass\n```", ("llvm",))

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json(path, {"accepted": False})
            self.assertEqual(json.loads(path.read_text()), {"accepted": False})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_gpu_probe(self):
        with patch("models.binary_decompilation.serve_when_free.subprocess.run") as command:
            command.return_value.stdout = "6, 80000, 81920\n7, 10000, 81920\n"
            self.assertEqual(gpu_memory()[6]["free_mib"], 80000)

    def test_active_gpu_jobs_are_not_treated_as_free(self):
        with patch("models.binary_decompilation.serve_when_free.subprocess.run") as command:
            command.side_effect = [subprocess.CompletedProcess([], 0, "6, GPU-six\n7, GPU-seven\n"),
                                   subprocess.CompletedProcess([], 0, "GPU-seven\n")]
            self.assertEqual(active_compute_gpus(), {7})

    def test_gpu_probe_propagates_failure(self):
        with patch("models.binary_decompilation.serve_when_free.subprocess.run",
                   side_effect=subprocess.CalledProcessError(9, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                gpu_memory()

    def test_basename_harness_baseline(self):
        config = json.loads(Path("models/binary_decompilation/configs/coreutils_qwen38.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            adapter = CoreutilsAdapter(config, "basename", Path(directory))
            tests = adapter.test(adapter.binary, Path(directory))
            self.assertEqual(tests["failed"], 0, tests)
            self.assertEqual(tests["passed"], len(config["programs"]["basename"]["cases"]) +
                             len(config["programs"]["basename"]["tests"]))


if __name__ == "__main__":
    unittest.main()
