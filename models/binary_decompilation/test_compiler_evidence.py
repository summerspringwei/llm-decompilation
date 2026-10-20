"""Native compiler evidence, diagnostics, and current-run reporting regressions."""

import json
from pathlib import Path
import tempfile
import unittest

from models.binary_decompilation.compiler_evidence import analyze_bytes, cfg_facts, ir_facts, relative_jump_tables
from models.binary_decompilation.public_abi import public_layouts
from models.binary_decompilation.validation_feedback import runtime_feedback
from models.binary_decompilation.test_native_validation import CLANG
from models.binary_decompilation.project_pipeline import write_json
from models.binary_decompilation.project_report import report


class CompilerEvidenceTests(unittest.TestCase):
    def test_bounded_relative_jump_table(self):
        import struct
        code = bytes.fromhex('83f9017710488d15f40f000048630c8a4801d1ffe1c3c3')
        # Without an upper-half proof, cmp ecx does not constrain rcx.
        self.assertEqual(relative_jump_tables(code, 0x1000, lambda a, n: struct.pack('<ii', -4075, -4074)), [])
        code = bytes.fromhex('89c983f9017710488d15f20f000048630c8a4801d1ffe1c3c3')
        def read(address, size):
            self.assertEqual((address, size), (0x2000, 8))
            return struct.pack('<ii', 0x1017 - 0x2000, 0x1018 - 0x2000)
        tables = relative_jump_tables(code, 0x1000, read)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]['entries'][1], {'index': 1, 'target': '0x1018'})
        facts = analyze_bytes(code, 0x1000, resolved_jumps={0x1015: [0x1017, 0x1018]})
        self.assertFalse(facts['unresolved_indirect_jumps'])
        self.assertEqual(len(relative_jump_tables(code, 0x1000, lambda a, n: b'\0' * n)), 0)

    def test_loop_dominators_ignore_unreachable_predecessor(self):
        blocks = [{"start": 'entry', "successors": ['loop']},
                  {"start": 'loop', "successors": ['body', 'exit']},
                  {"start": 'body', "successors": ['loop']},
                  {"start": 'dead', "successors": ['body']},
                  {"start": 'exit', "successors": []}]
        facts = cfg_facts(blocks, 'entry')
        self.assertEqual(facts['unreachable_blocks'], ['dead'])
        self.assertEqual(facts['natural_loops'][0]['blocks'], ['body', 'loop'])
        self.assertEqual(facts['immediate_dominators']['exit'], 'loop')

    def test_calls_are_not_cfg_edges_and_indirect_jumps_are_explicit(self):
        # call next; mov eax,[rdi+24]; test eax,eax; jne mov; jmp rax
        facts = analyze_bytes(bytes.fromhex('e8000000008b471885c075f9ffe0'), 0x1000)
        self.assertEqual(len(facts['calls']), 1)
        self.assertEqual(facts['unresolved_indirect_jumps'], ['0x100c'])
        self.assertEqual(facts['natural_loops'][0]['header'], '0x1005')
        access = facts['memory_accesses'][0]
        self.assertEqual((access['base'], access['offset'], access['width_bytes']), ('rdi', 24, 4))
        self.assertTrue(access['read'])

    def test_public_layouts_are_compiler_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = public_layouts(CLANG, Path(tmp))['records']
        self.assertGreaterEqual(result['struct stat']['size'], 32)
        self.assertEqual(result['struct stat']['offsets']['st_mode'], 24)
        self.assertEqual(result['struct option']['offsets']['has_arg'], 8)

    def test_llvm_normalization_and_analysis(self):
        ir = 'define i32 @leaf(i32 %x) {\nentry:\n %slot = alloca i32\n store i32 %x, ptr %slot\n %y = load i32, ptr %slot\n ret i32 %y\n}\n'
        with tempfile.TemporaryDirectory() as tmp:
            result = ir_facts(CLANG, ir, Path(tmp))
            self.assertTrue(result['available'], result)
            self.assertNotIn('alloca', result['normalized_ir'])
            self.assertIn('DominatorTree', result['dominator_and_loop_report'])
            self.assertEqual((Path(tmp) / 'accepted.ll').read_text(), ir)

    def test_invalid_ir_analysis_does_not_replace_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ir_facts(CLANG, 'not llvm', Path(tmp))
        self.assertFalse(result['available'])
        self.assertTrue(result['original_ir_unchanged'])

    def test_feedback_reports_loader_error_and_actual_coverage(self):
        tests = {'tests': [{'test': 'help', 'returncode': 1, 'expected_stdout': 'abcdef', 'stdout': 'abcXef'}]}
        events = [{'type': 'error', 'description': 'undefined symbol: wrong_global'}]
        details, text = runtime_feedback(tests, events, 'leaf')
        self.assertEqual(details['failure_kind'], 'loader_error')
        self.assertIn('wrong_global', text)
        self.assertEqual(details['counterexamples'][0]['stdout']['first_difference'], 3)
        self.assertEqual(details['coverage']['target_exercising_cases'], 0)
        events = [{'event': 'replacement', 'name': 'leaf', 'test_id': 'help'}] * 8
        details, _ = runtime_feedback({'tests': [{'test': 'help', 'returncode': 0}] * 14}, events, 'leaf')
        self.assertEqual(details['coverage']['suite_cases'], 14)
        self.assertEqual(details['coverage']['target_exercising_cases'], 1)

    def test_current_failed_result_overrides_historical_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'config.json', {'programs': {'fixture': {}}})
            d = root / 'retrieval/fixture'
            write_json(d / 'plan.json', {'targets': [{'name': 'leaf', 'function_id': 'leaf'}],
                                       'runtime': {'hits': {}}, 'instrumented_baseline': {}})
            for stage in ('signature', 'ir', 'c'):
                write_json(d / 'functions/leaf' / stage / 'latest.json', {'accepted': True})
            write_json(d / 'results.json', [{'name': 'leaf', 'signature': {'accepted': False, 'error': 'timeout'}}])
            row = report(root)['rows'][0]
        self.assertEqual(row['validated_ir'], 0)
        self.assertEqual(row['validated_c'], 0)


if __name__ == '__main__':
    unittest.main()
