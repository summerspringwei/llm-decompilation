"""Explicit source-scaffold comparison, never used by binary-first validation."""

from pathlib import Path
import re

from models.binary_decompilation.export_source_references import find_definition_span
from models.binary_decompilation.project_pipeline import CoreutilsAdapter, compiler_feedback, run, write_json


class SourceScaffoldAdapter(CoreutilsAdapter):
    binary_only = False
    requires_contract_header = True

    def __init__(self, config, program, output):
        super().__init__(config, program, output)
        self.header = ""
        import copy
        self.feedback_config = copy.deepcopy(config)
        self.feedback_config["programs"][program]["tests"] = []

    def feedback_test(self, executable, directory):
        return CoreutilsAdapter(self.feedback_config, self.program, directory).test(executable, directory)

    def compile_source(self, source, output, optimization="-O3"):
        # Gnulib's header-only inline helpers need optimization to avoid unresolved externals.
        return super().compile_source(source, output, "-O1" if optimization == "-O0" else optimization)

    def test(self, executable, directory):
        if getattr(self, "in_feedback", False):
            return self.feedback_test(executable, directory)
        return super().test(executable, directory)

    def validate(self, name, source_file, signature, candidate, directory, stage):
        if not source_file.endswith(".h"):
            self.in_feedback = True
            try:
                result = super().validate(name, source_file, signature, candidate, directory, stage)
            finally:
                self.in_feedback = False
            result["validation_mode"] = "source-scaffold-assisted"
            return result
        directory.mkdir(parents=True, exist_ok=True)
        original = (self.source / source_file).read_text()
        span = find_definition_span(original, name)
        if span is None:
            return {"accepted": False, "feedback": "Source helper definition missing"}
        replacement = candidate.read_text() if stage == "c" else re.sub(r"\bstatic\s+", "", signature)
        (directory / Path(source_file).name).write_text(original[:span[0]] + replacement + original[span[1]:])
        source = directory / "scaffold.c"
        source.write_text((self.source / "src" / (self.program + ".c")).read_text())
        scaffold = directory / "scaffold.o"
        compiled = self.compile_source(source, scaffold, "-O0" if stage == "ir" else "-O3")
        write_json(directory / "compile.json", compiled)
        if compiled["returncode"]:
            return {"accepted": False, "feedback": compiler_feedback(compiled["stderr"])}
        if stage == "signature":
            return {"accepted": True, "feedback": "Source scaffold accepted inferred declaration",
                    "validation_mode": "source-scaffold-assisted"}
        objects = [scaffold]
        if stage == "ir":
            obj = directory / "candidate.o"
            compiled = run([self.clang, "-Wno-override-module", "-c", str(candidate), "-o", str(obj)])
            if compiled["returncode"]:
                return {"accepted": False, "feedback": compiled["stderr"]}
            symbols = run(["nm", "--defined-only", str(scaffold)])["stdout"]
            local = [p[-1] for line in symbols.splitlines() if len(p := line.split()) == 3 and p[1] in "tdbr"]
            public = directory / "scaffold.public.o"
            result = run(["objcopy", *["--globalize-symbol=" + n for n in local], str(scaffold), str(public)])
            if result["returncode"]:
                return {"accepted": False, "feedback": result["stderr"]}
            objects = [public, obj]
        executable = directory / "recovered"
        linked = self.link(objects, executable, f"src/{self.program}.c")
        write_json(directory / "link.json", linked)
        if linked["returncode"]:
            return {"accepted": False, "feedback": linked["stderr"]}
        tests = self.feedback_test(executable, directory / "feedback")
        write_json(directory / "tests.json", tests)
        return {"accepted": tests["failed"] == 0 and tests["passed"] > 0,
                "feedback": str([t for t in tests["tests"] if t["returncode"] not in (0, 77)])[-12000:],
                "validation_mode": "source-scaffold-assisted", "tests": tests}
