"""Real PIE replacement tests; these are infrastructure, not model results."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from models.binary_decompilation.native_validation import build_candidate, make_wrapper, validate_probe
from models.binary_decompilation.project_pipeline import run


CLANG = "/data1/xiachunwei/Software/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang"
FIXTURE = '''#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
static int counter = 7;
__attribute__((noinline)) static int helper(int x) { return x * 2; }
__attribute__((noinline)) static int leaf(int x) { counter += x; return helper(x) + counter; }
__attribute__((noinline)) static int recurse(int x) { return x ? recurse(x-1)+1 : counter; }
__attribute__((noinline)) int tiny(int x) { return x; }
__attribute__((noinline)) int sum(int count, ...) {
 va_list ap; va_start(ap, count); int value=0;
 for(int i=0;i<count;i++) value+=va_arg(ap,int); va_end(ap); return value;
}
int main(int argc,char **argv) {
 int x = argc>1 ? atoi(argv[1]) : 3;
 printf("%d %d %d %d %d\\n",leaf(x),counter,recurse(3),tiny(x),sum(3,x,2,4));
 return 0;
}
'''


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "original.c"
        self.source.write_text(FIXTURE)
        self.binary = self.root / "original"
        result = run([CLANG, "-O1", "-fPIE", "-pie", str(self.source), "-o", str(self.binary)])
        self.assertEqual(result["returncode"], 0, result)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, name=None, code=None, header="", stage="c", args=None):
        manifests = []
        if name:
            candidate = self.root / ("candidate.ll" if stage == "ir" else "candidate.c")
            candidate.write_text(code)
            built = build_candidate(CLANG, self.binary, name, candidate, header, self.root / "build", stage)
            self.assertTrue(built["accepted"], built)
            manifests.append(built["manifest"])
        wrapper = make_wrapper(CLANG, self.binary, [name] if name else ["main"], manifests,
                               self.root / "runtime")
        actual = subprocess.run([str(wrapper), *(args or ["3"])], capture_output=True, timeout=30)
        expected = subprocess.run([str(self.binary), *(args or ["3"])], capture_output=True, timeout=30)
        events = [json.loads(line) for line in (self.root / "runtime/events.jsonl").read_text().splitlines()]
        return actual, expected, events

    def test_unmodified_trace_preserves_output_and_status(self):
        actual, expected, events = self.execute()
        self.assertEqual((actual.returncode, actual.stdout, actual.stderr),
                         (expected.returncode, expected.stdout, expected.stderr))
        self.assertTrue(any(e.get("event") == "entry" and e["name"] == "main" for e in events))

    def test_c_binds_private_mutable_data_and_callee(self):
        actual, expected, events = self.execute("leaf", "int leaf(int x) { counter += x; return helper(x)+counter; }",
                                                "extern int counter; int helper(int);")
        self.assertEqual((actual.returncode, actual.stdout, actual.stderr),
                         (expected.returncode, expected.stdout, expected.stderr))
        self.assertTrue(any(e.get("event") == "replacement" for e in events))

    def test_assembler_alias_preserves_dotted_global(self):
        self.source.write_text(FIXTURE.replace('static int counter = 7;',
                                               'static int counter __asm__("counter.flag") = 7;'))
        result = run([CLANG, '-O1', '-fPIE', '-pie', str(self.source), '-o', str(self.binary)])
        self.assertEqual(result['returncode'], 0, result)
        actual, expected, _ = self.execute('leaf', 'int leaf(int x) { counter += x; return helper(x)+counter; }',
                                           'extern int counter __asm__("counter.flag"); int helper(int);')
        self.assertEqual((actual.returncode, actual.stdout, actual.stderr),
                         (expected.returncode, expected.stdout, expected.stderr))

    def test_loader_error_is_not_hidden_by_missing_ping(self):
        actual, _, events = self.execute('leaf', 'int leaf(int x) { return unknown_fixture_global+x; }',
                                         'extern int unknown_fixture_global;')
        self.assertEqual(actual.returncode, 125)
        self.assertIn(b'undefined symbol: unknown_fixture_global', actual.stderr)
        self.assertNotIn(b"unable to find method 'ping'", actual.stderr)
        self.assertTrue(any('unknown_fixture_global' in e.get('description', '') for e in events))

    def test_ir_live_global_binding(self):
        ir = '''@counter = external global i32
declare i32 @helper(i32)
define i32 @leaf(i32 %x) {
 %old = load i32, ptr @counter
 %new = add i32 %old, %x
 store i32 %new, ptr @counter
 %h = call i32 @helper(i32 %x)
 %y = add i32 %h, %new
 ret i32 %y
}
'''
        actual, expected, events = self.execute("leaf", ir, stage="ir")
        self.assertEqual((actual.returncode, actual.stdout, actual.stderr),
                         (expected.returncode, expected.stdout, expected.stderr))

    def test_incorrect_candidate_fails_differential(self):
        actual, expected, events = self.execute("leaf", "int leaf(int x) { return x + 99; }")
        self.assertNotEqual(actual.stdout, expected.stdout)

    def test_crash_feedback_is_delivered_before_process_exit(self):
        actual, expected, events = self.execute("leaf", "int leaf(int x) { return *(volatile int *)3; }")
        self.assertNotEqual(actual.returncode, 0)
        self.assertTrue(any(e.get("event") == "crash" for e in events), events)

    def test_infinite_candidate_is_timed_out_and_reaped(self):
        candidate = self.root / "loop.ll"
        candidate.write_text("define i32 @leaf(i32 %x) { entry: br label %loop\nloop: br label %loop\n}\n")
        built = build_candidate(CLANG, self.binary, "leaf", candidate, "", self.root / "loop_build", "ir")
        self.assertTrue(built["accepted"], built)
        wrapper = make_wrapper(CLANG, self.binary, [], [built["manifest"]], self.root / "loop_runtime", runtime_timeout=1)
        result = subprocess.run([str(wrapper)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 124, result.stderr)
        events = [json.loads(line) for line in (self.root / "loop_runtime/events.jsonl").read_text().splitlines()]
        timeout = next(e for e in events if e.get("event") == "timeout")
        with self.assertRaises(ProcessLookupError):
            os.kill(timeout["pid"], 0)

    def test_probe_schema_rejects_bad_types_refs_and_allocations(self):
        for probe in (
            {"return_type": "int", "argument_types": ["integer"], "arguments": [0]},
            {"return_type": "int", "argument_types": ["pointer"], "arguments": [{"ref": "global_name"}]},
            {"return_type": "int", "argument_types": ["pointer"], "arguments": [{"buffer": 8, "bytes": [1] * 9}]},
            {"return_type": [], "argument_types": [], "arguments": []},
        ):
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                validate_probe(probe)

    def test_probe_setup_allocations_can_be_referenced_by_target(self):
        validate_probe({"return_type": "int", "argument_types": ["pointer"], "arguments": [{"ref": "stat_data"}],
                        "setup": [{"symbol": "stat", "return_type": "int", "argument_types": ["pointer", "pointer"],
                                   "arguments": [{"string": "/"}, {"buffer": 144, "id": "stat_data"}]}],
                        "memory_outputs": [{"id": "stat_data", "size": 128}]})

    def test_recursion(self):
        actual, expected, events = self.execute("recurse", "int recurse(int x) { return x ? recurse(x-1)+1 : counter; }",
                                                "extern int counter;")
        self.assertEqual(actual.stdout, expected.stdout, actual.stderr)
        self.assertEqual(actual.returncode, 0, actual.stderr)

    def test_tiny_function(self):
        actual, expected, events = self.execute("tiny", "int tiny(int x) { return x; }")
        self.assertEqual((actual.returncode, actual.stdout), (0, expected.stdout), actual.stderr)

    def test_tiny_ir_candidate_has_safe_instrumentation_entry(self):
        candidate = self.root / "tiny.ll"
        candidate.write_text("define i32 @tiny(i32 %x) { ret i32 %x }")
        built = build_candidate(CLANG, self.binary, "tiny", candidate, "", self.root / "tiny_build", "ir")
        self.assertTrue(built["accepted"], built)
        probe = {"name": "tiny", "return_type": "int", "argument_types": ["int"], "arguments": [11]}
        baseline = make_wrapper(CLANG, self.binary, [], [], self.root / "tiny_original", direct_probe=probe)
        recovered = make_wrapper(CLANG, self.binary, [], [built["manifest"]], self.root / "tiny_runtime", direct_probe=probe)
        expected = subprocess.run([str(baseline)], capture_output=True, timeout=10)
        actual = subprocess.run([str(recovered)], capture_output=True, timeout=10)
        self.assertEqual((actual.returncode, actual.stdout), (0, expected.stdout), actual.stderr)
        events = [json.loads(line) for line in (self.root / "tiny_runtime/events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e.get("event") == "replacement" for e in events))

    def test_hanging_module_constructor_is_timed_out_and_reaped(self):
        candidate = self.root / "constructor.c"
        candidate.write_text('__attribute__((constructor)) static void hang(void) { for(;;) __asm__ volatile(""); }\n'
                             'int leaf(int x) { return x; }\n')
        built = build_candidate(CLANG, self.binary, "leaf", candidate, "", self.root / "constructor_build", "c")
        self.assertTrue(built["accepted"], built)
        wrapper = make_wrapper(CLANG, self.binary, [], [built["manifest"]], self.root / "constructor_runtime", runtime_timeout=1)
        result = subprocess.run([str(wrapper)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 124, result.stderr)
        events = [json.loads(line) for line in (self.root / "constructor_runtime/events.jsonl").read_text().splitlines()]
        timeout = next(e for e in events if e.get("event") == "timeout")
        with self.assertRaises(ProcessLookupError):
            os.kill(timeout["pid"], 0)

    def test_variadic_native_abi(self):
        actual, expected, events = self.execute("sum", '''int sum(int count, ...) {
 va_list ap; va_start(ap,count); int total=0;
 for(int i=0;i<count;i++) total+=va_arg(ap,int); va_end(ap); return total;
}''', "#include <stdarg.h>")
        self.assertEqual((actual.returncode, actual.stdout), (0, expected.stdout), actual.stderr)

    def test_variadic_dependency_bridge_preserves_argument_registers(self):
        actual, expected, events = self.execute("leaf", "int leaf(int x) { counter += x; return sum(2,helper(x),counter); }",
                                               "extern int counter; int helper(int); int sum(int,...);")
        self.assertEqual((actual.returncode, actual.stdout), (0, expected.stdout), actual.stderr)

    def test_direct_probe_bypasses_application_entry(self):
        candidate = self.root / "direct.c"
        candidate.write_text("int leaf(int x) { counter += x; return helper(x)+counter; }")
        built = build_candidate(CLANG, self.binary, "leaf", candidate, "extern int counter; int helper(int);", self.root / "direct_build", "c")
        self.assertTrue(built["accepted"], built)
        probe = {"name": "leaf", "return_type": "int", "argument_types": ["int"],
                 "arguments": [11], "return_kind": "integer"}
        baseline = make_wrapper(CLANG, self.binary, [], [], self.root / "direct_original", direct_probe=probe)
        recovered = make_wrapper(CLANG, self.binary, [], [built["manifest"]], self.root / "direct_recovered", direct_probe=probe)
        expected = subprocess.run([str(baseline)], capture_output=True, timeout=30)
        actual = subprocess.run([str(recovered)], capture_output=True, timeout=30)
        self.assertEqual((expected.returncode, actual.returncode), (0, 0), (expected.stderr, actual.stderr))
        self.assertEqual(expected.stdout, actual.stdout)
        self.assertIn(b'"40"', actual.stdout)


if __name__ == "__main__":
    unittest.main()
