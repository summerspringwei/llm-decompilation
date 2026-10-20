"""Actual generated-only build checks; fixtures are not benchmark predictions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from models.binary_decompilation.project_pipeline import CoreutilsAdapter, run
from models.binary_decompilation.standalone_recovery import (
    binary_data_evidence, build_generated_project, generated_includes_only, support_dependencies,
    recover_standalone,
)


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        self.config = json.loads(Path("models/binary_decompilation/configs/coreutils_qwen38.json").read_text())
        self.config.update(source_dir=str(self.root), build_dir=str(self.root), programs={
            "fixture": {"tests": [], "cases": [{"args": ["6"]}, {"args": ["-3"]}],
                        "standalone": {"support_objects": []}}})
        self.adapter = CoreutilsAdapter(self.config, "fixture", self.root)
        self.header = "#include <stdio.h>\n#include <stdlib.h>\nextern int offset;\nint transform(int);\nint main(int, char **);\n"
        self.globals = "int offset = 3;\n"
        self.functions = []
        bodies = {
            "transform": "int transform(int x) { return x + offset; }\n",
            "main": 'int main(int argc, char **argv) { printf("%d\\n", transform(atoi(argv[1]))); return 0; }\n',
        }
        modules = {
            "transform": '@offset = external global i32\ndefine i32 @transform(i32 %x) {\n'
                         '%o = load i32, ptr @offset\n%y = add i32 %x, %o\nret i32 %y\n}\n',
            "main": '@fmt = private constant [4 x i8] c"%d\\0A\\00"\n'
                    'declare i32 @atoi(ptr)\ndeclare i32 @printf(ptr, ...)\ndeclare i32 @transform(i32)\n'
                    'define i32 @main(i32 %argc, ptr %argv) {\n'
                    '%slot = getelementptr ptr, ptr %argv, i64 1\n%arg = load ptr, ptr %slot\n'
                    '%x = call i32 @atoi(ptr %arg)\n%y = call i32 @transform(i32 %x)\n'
                    '%r = call i32 (ptr, ...) @printf(ptr @fmt, i32 %y)\nret i32 0\n}\n',
        }
        for name in bodies:
            c, ir = self.root / (name + ".c"), self.root / (name + ".ll")
            c.write_text(bodies[name])
            ir.write_text(modules[name])
            self.functions.append({"name": name, "c_candidate": str(c), "ir_candidate": str(ir)})
        original = self.root / "original.c"
        original.write_text(self.header + self.globals + "".join(bodies.values()))
        built = run([self.adapter.clang, str(original), "-o", str(self.adapter.binary)])
        self.assertEqual(built["returncode"], 0, built)
        built = run([self.adapter.clang, "-c", str(original), "-o", str(self.root / "src/fixture.o")])
        self.assertEqual(built["returncode"], 0, built)
        original.unlink()

    def test_generated_c_and_ir_share_recovered_data_without_source(self):
        for kind in ("c", "ir"):
            result = build_generated_project(self.adapter, self.functions, self.header, self.globals,
                                             self.root / kind, kind)
            self.assertTrue(result["accepted"], result)
            self.assertFalse(result["provenance"]["application_source_scaffold"])
            self.assertEqual(result["provenance"]["support_dependencies"], [])

    def test_incorrect_global_initialization_fails_runtime_checks(self):
        result = build_generated_project(self.adapter, self.functions, self.header, "int offset = 4;\n",
                                         self.root / "wrong-data", "c")
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["tests"]["failed"], 2)

    def test_missing_function_does_not_fall_back_to_original(self):
        function = next(f for f in self.functions if f["name"] == "transform")
        Path(function["c_candidate"]).write_text("/* int transform(int x) { return x; } */\n")
        result = build_generated_project(self.adapter, self.functions, self.header, self.globals,
                                         self.root / "missing", "c")
        self.assertFalse(result["accepted"], result)

    def test_original_application_object_is_forbidden_support(self):
        self.config["programs"]["fixture"]["standalone"]["support_objects"] = ["src/fixture.o"]
        with self.assertRaises(ValueError):
            support_dependencies(self.adapter)

    def test_glue_cannot_supply_target_functions(self):
        header = self.header + "int transform(int x) { return x + 3; }\n"
        result = build_generated_project(self.adapter, self.functions, header, self.globals,
                                         self.root / "glue-substitution", "c")
        self.assertFalse(result["accepted"], result)
        self.assertIn("Glue must not substitute", result["feedback"])

    def test_dependency_cannot_supply_application_definitions(self):
        copy = self.root / "disguised-support.o"
        copy.write_bytes((self.root / "src/fixture.o").read_bytes())
        self.config["programs"]["fixture"]["standalone"]["support_objects"] = [str(copy)]
        result = build_generated_project(self.adapter, self.functions, self.header, self.globals,
                                         self.root / "delegation", "c")
        self.assertFalse(result["accepted"], result)
        self.assertIn("defines application target", result["feedback"])

    def test_original_sources_cannot_be_included(self):
        for text in ('#include "/tmp/original.c"', '#include "../original.h"', '#include ORIGINAL_HEADER'):
            with self.assertRaises(ValueError):
                generated_includes_only(text)

    def test_binary_global_evidence_is_source_independent(self):
        evidence = binary_data_evidence(self.adapter.binary, self.root / "src/fixture.o", [])
        offset = next(obj for obj in evidence["objects"] if obj["name"] == "offset")
        self.assertEqual(offset["initial_bytes_hex"], "03000000")
        self.assertNotIn("c_type", offset)

    def integration_inputs(self):
        self.config["max_attempts"] = 1
        results = [{"name": f["name"], "signature": {"signature": "int " + f["name"] + "();"},
                    "ir": {"accepted": True, "candidate": f["ir_candidate"]},
                    "c": {"accepted": True, "candidate": f["c_candidate"]}} for f in self.functions]
        plan = {"targets": [{"name": f["name"]} for f in self.functions], "binary_data": {"objects": []}}
        return plan, results

    def test_integration_revalidation_keeps_assisted_replacements(self):
        plan, results = self.integration_inputs()
        glue = {"project_header": self.header, "globals_c": self.globals,
                "function_replacements": {"transform": "int transform(int x) { return offset + x; }"}}
        response = {"choices": [{"message": {"content": json.dumps(glue)}, "finish_reason": "stop"}]}
        directory = self.root / "integration"
        with patch("models.binary_decompilation.standalone_recovery.fit_prompt", return_value=("fixture", {})), \
             patch("models.binary_decompilation.standalone_recovery.generate", return_value=response):
            first = recover_standalone(Mock(), self.config, self.adapter, plan, results, directory)
        self.assertTrue(first["accepted"], first)
        revised = next(f for f in first["generated_functions"] if f["name"] == "transform")["c_candidate"]
        self.assertNotEqual(revised, self.functions[0]["c_candidate"])
        with patch("models.binary_decompilation.standalone_recovery.generate", side_effect=AssertionError("No new request")):
            cached = recover_standalone(Mock(), self.config, self.adapter, plan, results, directory)
        self.assertTrue(cached["accepted"], cached)
        paths = [p["path"] for p in cached["validation"]["c"]["provenance"]["generated_inputs"]]
        self.assertIn(revised, paths)

    def test_incomplete_integration_response_never_builds(self):
        plan, results = self.integration_inputs()
        response = {"choices": [{"message": {"content": json.dumps({"project_header": self.header,
                    "globals_c": self.globals, "function_replacements": {}})}, "finish_reason": "length"}]}
        with patch("models.binary_decompilation.standalone_recovery.fit_prompt", return_value=("fixture", {})), \
             patch("models.binary_decompilation.standalone_recovery.generate", return_value=response), \
             patch("models.binary_decompilation.standalone_recovery.build_generated_project") as builder:
            result = recover_standalone(Mock(), self.config, self.adapter, plan, results, self.root / "incomplete")
        self.assertFalse(result["accepted"], result)
        builder.assert_not_called()

    def test_integration_uses_effective_completion_budget(self):
        plan, results = self.integration_inputs()
        self.config.update(max_tokens=260000, max_tokens_policy="remaining_context")
        response = {"choices": [{"message": {"content": "unfinished"}, "finish_reason": "length"}]}
        with patch("models.binary_decompilation.standalone_recovery.fit_prompt",
                   return_value=("fixture", {"max_output_tokens": 242527})), \
             patch("models.binary_decompilation.standalone_recovery.generate", return_value=response) as generate:
            recover_standalone(Mock(), self.config, self.adapter, plan, results, self.root / "budget")
        self.assertEqual(generate.call_args.args[1]["max_tokens"], 242527)
        self.assertEqual(generate.call_args.args[1]["requested_max_tokens"], 260000)


if __name__ == "__main__":
    unittest.main()
