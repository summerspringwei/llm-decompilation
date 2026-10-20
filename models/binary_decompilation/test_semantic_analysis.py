"""Regression fixtures for analysis facts and deliberately wrong candidates."""

import json
from pathlib import Path
import tempfile
import unittest

from models.binary_decompilation.compiler_evidence import analyze_bytes
from models.binary_decompilation.semantic_analysis import lift, dataflow, affine_memory, backward_slice, evaluate
from models.binary_decompilation.translation_contract import inspect_ir, llvm_bytes
from models.binary_decompilation.semantic_validation import supported_leaf, compare_scalar
from models.binary_decompilation.candidate_search import validation_rank
from models.binary_decompilation.project_pipeline import run
from models.binary_decompilation.test_native_validation import CLANG
from models.binary_decompilation.validation_feedback import runtime_feedback


def analyze(hex_code):
    code = bytes.fromhex(hex_code)
    records, errors, registers = lift(code, 0x1000)
    assert not errors, errors
    cfg = analyze_bytes(code, 0x1000)
    return records, registers, cfg, dataflow(records, cfg, registers)


class SemanticAnalysisTests(unittest.TestCase):
    def test_partial_register_and_return_slice(self):
        records, regs, cfg, flow = analyze('89f801f0c3')
        self.assertTrue(flow['converged'])
        self.assertEqual(flow['entry_argument_registers_read'], ['RDI', 'RSI'])
        ret = next(r for r in records if r['opcode'] == 'RETURN')
        self.assertTrue(any(s.startswith('0x1002:') for s in flow['observable_slices'][ret['id']]['definitions']))
        self.assertTrue(any(r['opcode'] == 'INT_ZEXT' and r['output'].get('name') == 'RAX' for r in records))

    def test_known_high_bytes_after_zero_extension(self):
        records, regs, cfg, flow = analyze('89f848c1e820c3')
        shift = next(r for r in records if r['opcode'] == 'INT_RIGHT' and r['output'].get('name') == 'RAX')
        self.assertIn(shift['id'], flow['constant_observations'])
        self.assertFalse(any(r['opcode'] == 'CALLOTHER' for r in records))

    def test_flags_dependencies_cross_block(self):
        records, regs, cfg, flow = analyze('85ff7406b801000000c331c0c3')
        branch = next(r for r in records if r['opcode'] == 'CBRANCH')
        self.assertTrue(any(s.startswith('0x1000:') for s in flow['observable_slices'][branch['id']]['definitions']))
        self.assertEqual(len(cfg['blocks']), 3)

    def test_loop_fixed_point(self):
        records, regs, cfg, flow = analyze('31c085ff740601f883ef0175f9c3')
        self.assertTrue(flow['converged'])
        self.assertGreater(flow['iterations'], 1)

    def test_call_clobber_is_not_old_definition(self):
        records, regs, cfg, flow = analyze('b807000000e80001000083c001c3')
        add = next(r for r in records if r['address'] == '0x100a' and r['opcode'] == 'INT_ADD')
        self.assertIsNone(flow['constant_observations'][add['id']]['inputs'][0])

    def test_affine_argument_and_stack(self):
        records, regs, cfg, _ = analyze('4883ec1048897c2408488b47184883c410c3')
        memory = affine_memory(records, cfg, regs)
        self.assertTrue(memory['converged'])
        expressions = [a['address_expression'] for a in memory['accesses']]
        self.assertIn({'base': 'entry:RDI', 'offset': 24}, expressions)
        self.assertIn({'base': 'entry:RSP', 'offset': -8}, expressions)

    def test_unknown_indirect_is_not_complete(self):
        _, _, _, flow = analyze('ffe0')
        self.assertEqual(flow['status'], 'unknown')

    def test_modular_constants(self):
        self.assertEqual(evaluate('INT_ADD', [255, 1], [1, 1], 1), 0)
        self.assertEqual(evaluate('INT_SEXT', [255], [1], 8), 2**64 - 1)
        self.assertIsNone(evaluate('FLOAT_ADD', [1, 2], [4, 4], 4))

    def test_cycle_safe_bounded_slice(self):
        self.assertEqual(backward_slice(['a'], {'a': ['b'], 'b': ['a']})['definitions'], ['a', 'b'])
        self.assertTrue(backward_slice(['a'], {'a': ['b']}, limit=1)['truncated'])

    def test_attributes_and_literals(self):
        self.assertFalse(inspect_ir('define i32 @f(i32 %x) { %v = add nsw i32 %x, 1\n ret i32 %v }')['accepted'])
        self.assertTrue(inspect_ir('; nsw\n@str = constant [4 x i8] c"nuw\\00"\ndefine void @noalias() { ret void }')['accepted'])
        self.assertEqual(llvm_bytes(b'a\n\0'), 'c"\\61\\0A\\00"')
        contract = {'optional_literal_definitions': [{'symbol': '__decomp_literal_1',
                    'definition': '@__decomp_literal_1 = private constant [2 x i8] c"\\61\\00", align 1'}]}
        self.assertFalse(inspect_ir('@__decomp_literal_1 = private constant [1 x i8] zeroinitializer', contract)['accepted'])

    def test_unrelated_tests_do_not_win_candidate_ranking(self):
        a = {'execution_observed': True, 'tests': {'passed': 100, 'failed': 1},
             'coverage': {'target_passed_cases': 0, 'target_failed_cases': 1}}
        b = {'execution_observed': True, 'tests': {'passed': 2, 'failed': 8},
             'coverage': {'target_passed_cases': 2, 'target_failed_cases': 0}}
        self.assertGreater(validation_rank(b), validation_rank(a))

    def test_direct_probe_counterexample(self):
        tests = {'tests': [{'test': 'direct_probe:0', 'target_executed': True, 'returncode': 1,
                          'expected': {'returncode': 0, 'stdout': '12'},
                          'actual': {'returncode': 0, 'stdout': '13'}}]}
        details, _ = runtime_feedback(tests, [], 'f')
        self.assertEqual(details['counterexamples'][0]['stdout']['first_difference'], 1)
        self.assertEqual(details['coverage']['target_failed_cases'], 1)

    def test_compiler_slice_is_advisory(self):
        from models.binary_decompilation.repair_evidence import compiler_slice
        records, _, cfg, flow = analyze('89f801f0c3')
        facts = {'pcode': records, 'cfg': cfg, 'dataflow': flow}
        ir = '; asm-block: 0x1000\n%bad = 1\n'
        sliced = compiler_slice(facts, ir, '/tmp/candidate.ll:2:1: error: expected instruction')
        self.assertTrue(sliced['localized'])
        self.assertEqual(sliced['status'], 'advisory_model_mapping_not_proof')
        self.assertFalse(compiler_slice(facts, ir, 'stdout mismatch')['localized'])

    def test_infrastructure_is_not_semantic_failure(self):
        details, _ = runtime_feedback({'tests': [{'test': 'x', 'returncode': 1,
                'stderr': 'Binary isolation infrastructure failed: ptrace denied'}]}, [], 'f')
        self.assertEqual(details['failure_kind'], 'infrastructure')

    def test_semantic_checker_rejects_memory_and_calls(self):
        self.assertIn('Memory', supported_leaf(bytes.fromhex('8b07c3'), 0x1000))
        self.assertIn('Calls', supported_leaf(bytes.fromhex('e800000000c3'), 0x1000))

    def test_scalar_equivalence_and_concrete_mutant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            libraries = []
            for index, body in enumerate(('return x + 1;', 'return 1 + x;', 'return x + 2;')):
                source, library = root / f'{index}.c', root / f'{index}.so'
                source.write_text('unsigned f(unsigned x) { ' + body + ' }\n')
                compiled = run([CLANG, '-O2', '-shared', '-fPIC', str(source), '-o', str(library)])
                self.assertEqual(compiled['returncode'], 0, compiled)
                libraries.append(library)
            same = compare_scalar(libraries[0], libraries[1], 'f', return_bits=32)
            different = compare_scalar(libraries[0], libraries[2], 'f', return_bits=32)
            self.assertEqual(same['status'], 'bounded_equivalent', same)
            self.assertEqual(different['status'], 'counterexample', different)
            self.assertNotEqual(different['expected_return'], different['actual_return'])


if __name__ == '__main__':
    unittest.main()
