"""Budget, ownership, and evidence checks for the reusable campaign."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.binary_decompilation.binary_project import BinaryCoreutilsAdapter, load_config, normalize_ghidra_addresses, run_stage, summarize_trace
from models.binary_decompilation.project_pipeline import generate, write_json
from models.binary_decompilation.project_interfaces import resolve_factory
from models.binary_decompilation.assisted_validation import SourceScaffoldAdapter


class BinaryProjectTests(unittest.TestCase):
    def test_parallel_workers_keep_checkpoints_and_merge_results(self):
        import argparse
        from models.binary_decompilation.binary_project import parallel_programs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = ["basename", "uniq", "df"]
            config = {"programs": dict.fromkeys(programs, {}), "max_attempts": 10, "num_generate": 8}
            args = argparse.Namespace(config="fixture.json", function=None, prepare_only=False)
            launched = []

            def launch(command, **kwargs):
                name = command[command.index("--program") + 1]
                metadata = Path(command[command.index("--metadata-dir") + 1])
                self.assertEqual(metadata, root / "workers" / name)
                self.assertNotIn("--parallel-programs", command)
                write_json(metadata / "campaign.json", {"retrieval/" + name: {"complete_project_recovery": True}})
                process = Mock(pid=100 + len(launched))

                def poll():
                    self.assertEqual(len(launched), 3)
                    return 0

                process.poll.side_effect = poll
                launched.append(name)
                return process

            checkpoint = root / "retrieval/basename/functions/example/ir/latest.json"
            write_json(checkpoint, {"accepted": True})
            with patch("models.binary_decompilation.binary_project.subprocess.Popen", side_effect=launch):
                parallel_programs(args, config, programs, ["retrieval"], root)
            self.assertEqual(len(json.loads((root / "campaign.json").read_text())), 3)
            self.assertTrue(json.loads((root / "acceptance.json").read_text())["goal_complete"])
            self.assertEqual(json.loads(checkpoint.read_text()), {"accepted": True})

    def test_assisted_ir_scaffold_enables_header_inline_helpers(self):
        adapter = object.__new__(SourceScaffoldAdapter)
        with patch("models.binary_decompilation.project_pipeline.CoreutilsAdapter.compile_source") as compile_source:
            adapter.compile_source(Path("source.c"), Path("source.o"), "-O0")
            self.assertEqual(compile_source.call_args.args[-1], "-O1")
            adapter.compile_source(Path("source.c"), Path("source.o"), "-O3")
            self.assertEqual(compile_source.call_args.args[-1], "-O3")

    def test_selected_resume_preserves_other_campaign_summaries(self):
        from models.binary_decompilation.binary_project import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = {"retrieval/fixture": {"complete_project_recovery": True}}
            write_json(root / "campaign.json", previous)
            config = {"num_generate": 1, "max_attempts": 10, "output_dir": tmp,
                      "programs": {"fixture": {}}, "clang": "fixture-clang"}
            with patch("sys.argv", ["binary_project", "--config", "fixture.json", "--track", "assisted"]), \
                 patch("models.binary_decompilation.binary_project.load_config", return_value=config), \
                 patch("models.binary_decompilation.binary_project.model_client", return_value=Mock()), \
                 patch("models.binary_decompilation.service_preflight.check_services", return_value={"ready": True}), \
                 patch("models.binary_decompilation.binary_project.run", return_value={"returncode": 0}), \
                 patch("models.binary_decompilation.binary_project.run_program", return_value={"complete_project_recovery": False}):
                main()
            campaign = json.loads((root / "campaign.json").read_text())
            self.assertEqual(campaign["retrieval/fixture"], previous["retrieval/fixture"])
            self.assertIn("assisted/fixture", campaign)
            self.assertTrue(json.loads((root / "acceptance.json").read_text())["goal_complete"])

    def test_vector_collection_configuration_matches_hermessim(self):
        config = load_config(Path("models/binary_decompilation/configs/coreutils_binary_first.json"))
        self.assertEqual(config["retrieval"]["expected_dimension"], 384)
        self.assertTrue(config["retrieval"]["collection_template"].endswith("_hermessim"))
        self.assertEqual(config["num_generate"], 1)
        self.assertEqual(config["max_attempts"], 10)
        self.assertEqual(config["programs"]["df"]["cases"][-1]["tty"], True)

    def test_ghidra_addresses_are_normalized_not_stack_offsets(self):
        evidence = {"entry": "001027d0", "cfg": [{"start": "001027d0", "end": "001027df"}],
                    "references": [{"address": "Stack[-0x50]"}]}
        result = normalize_ghidra_addresses(evidence, 0x100000)
        self.assertEqual(result["entry"], "0x27d0")
        self.assertEqual(result["references"][0]["address"], "Stack[-0x50]")

    def test_trace_keeps_unobserved_targets_and_test_identity(self):
        events = [{"event": "entry", "name": "leaf", "registers": {"rdi": "0x1"},
                   "stack": ["leaf", "main"], "test_id": "probe"}]
        result = summarize_trace(events, [{"name": "leaf"}, {"name": "main"}, {"name": "unseen"}])
        self.assertEqual(result["hits"]["unseen"], 0)
        self.assertIn(("main", "leaf"), result["observed_edges"])
        self.assertEqual(result["test_ids"], ["probe"])
        self.assertEqual(result["entry_register_samples"], {})

    def test_changed_inputs_share_budget_including_interrupted_requests(self):
        config = {"max_attempts": 10, "model": "fixture", "base_url": "http://fixture/v1",
                  "max_tokens": 128, "context_window": 4096}
        adapter = Mock(header="", program="fixture", feedback_config={"programs": {"fixture": {"cases": []}}})
        function = {"name": "leaf", "function_id": "hash:0x10", "assembly": "ret"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for attempt in range(9):
                write_json(directory / "old_revision" / f"attempt_{attempt}" / "request.json", {"state": "started"})
            def infer(client, settings, *args):
                revision = args[4]
                self.assertEqual(settings["max_attempts"], 1)
                write_json(revision / "attempt_0/request.json", {"state": "started"})
                return {"accepted": False}
            with patch("models.binary_decompilation.binary_project.infer_stage", side_effect=infer):
                first = run_stage(Mock(), config, adapter, function, {}, "ir", directory)
            self.assertEqual(first["total_generation_requests"], 10)
            second = run_stage(Mock(), config, adapter, function, {"changed": True}, "ir", directory)
            self.assertEqual(second["reason"], "stage generation budget exhausted")
            self.assertEqual(json.loads((directory / "latest.json").read_text()), second)

    def test_feedback_and_upstream_runtime_evidence_are_separate(self):
        events = [{"event": "entry", "name": "leaf", "registers": {"rdi": "0x1"},
                   "stack": ["leaf"], "test_id": test_id} for test_id in
                  ("tests/fixture.sh", "differential:fixture")]
        result = summarize_trace(events, [{"name": "leaf"}])
        self.assertEqual(result["hits"]["leaf"], 2)
        self.assertEqual(len(result["entry_register_samples"]["leaf"]), 1)
        self.assertEqual(result["samples"]["leaf"][0]["test_id"], "differential:fixture")

    def test_cached_signature_revalidation_reads_original_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"source_dir": tmp, "build_dir": tmp, "clang": "fixture-clang",
                      "programs": {"fixture": {"tests": [], "cases": []}}}
            adapter = BinaryCoreutilsAdapter(config, "fixture", root)
            candidate = root / "attempt_0/unused.c"
            header = "int leaf(int);\n"
            write_json(candidate.parent / "contract.json", {"project_header": header, "direct_probes": []})
            compiled = {"returncode": 0, "stdout": json.dumps({"inner": [
                {"kind": "FunctionDecl", "name": "leaf", "inner": []}]}), "stderr": ""}
            with patch("models.binary_decompilation.binary_project.run", return_value=compiled):
                for directory in (candidate.parent / "validation", root / "revalidation"):
                    result = adapter.validate("leaf", "fixture.c", "int leaf(int);", candidate, directory, "signature")
                    self.assertTrue(result["accepted"], result)
                    self.assertEqual(adapter.header, header)
                    self.assertFalse(result["abi_verified"])

    def test_streaming_accumulates_one_completion_and_usage(self):
        client = Mock()
        chunks = []
        for content, finish in [("abc", None), ("def", "stop")]:
            chunk = Mock()
            chunk.model_dump.return_value = {"choices": [{"delta": {"content": content}, "finish_reason": finish}], "usage": None}
            chunks.append(chunk)
        client.chat.completions.create.return_value = iter(chunks)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(client, {"model": "fixture", "stream": True}, "prompt", Path(tmp))
        self.assertEqual(result["choices"][0]["message"]["content"], "abcdef")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["n"], 1)

    def test_adapter_factory_requires_explicit_module_and_callable(self):
        self.assertTrue(callable(resolve_factory("models.binary_decompilation.binary_project:BinaryCoreutilsAdapter")))
        with self.assertRaises(ValueError):
            resolve_factory("missing_factory")

    def test_streaming_stops_repetitive_empty_constants(self):
        client = Mock()
        stream = Mock()
        chunks = []
        for index in range(100):
            chunk = Mock()
            content = f'@.str.{index} = private constant [1 x i8] c"\\00", align 1\n'
            chunk.model_dump.return_value = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
            chunks.append(chunk)
        stream.__iter__ = Mock(return_value=iter(chunks))
        client.chat.completions.create.return_value = stream
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(client, {"model": "fixture", "stream": True}, "prompt", Path(tmp))
        self.assertEqual(result["choices"][0]["finish_reason"], "degenerate_constant_repetition")
        stream.close.assert_called_once()

    def test_streaming_stops_long_repetitive_allocas(self):
        client = Mock()
        stream = Mock()
        chunks = []
        for index in range(100):
            chunk = Mock()
            content = f'  %a.long.variable.name.{index} = alloca [128 x i64], align 16\n'
            chunk.model_dump.return_value = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
            chunks.append(chunk)
        stream.__iter__ = Mock(return_value=iter(chunks))
        client.chat.completions.create.return_value = stream
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(client, {"model": "fixture", "stream": True}, "prompt", Path(tmp))
        self.assertEqual(result["choices"][0]["finish_reason"], "degenerate_alloca_repetition")
        stream.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
