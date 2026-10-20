"""Native ABI parsing, worklist propagation, baseline isolation, and request budgets."""

import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from models.binary_decompilation.signature_reconciliation import (
    ABIReader, ReconciliationAdapter, abi_identity, contradictions, reconcile_worklist,
    extract_call_sites,
)
from models.binary_decompilation.reconciliation_evidence import raw_evidence, assert_baseline_boundary, pack
from models.binary_decompilation.reconciled_campaign import affected_callers, batch, commit, attach_abi
from models.binary_decompilation.project_pipeline import run, write_json
from models.binary_decompilation.test_native_validation import CLANG


def abi(result='i32', parameters=('i32',)):
    return {'calling_convention': 0, 'defined': False, 'parameters': [{'type': p, 'attributes': []} for p in parameters],
            'return_type': result, 'return_attributes': [], 'variadic': False}


def contract(name, result='i32', dependencies=None):
    return {'accepted': True, 'signature': name, 'abi': abi(result), 'dependencies': dependencies or {},
            'contract': {'project_header': 'extern int ' + name + '(int);'}}


class ReconciliationLogicTests(unittest.TestCase):
    def test_detects_both_ends_of_call_edge(self):
        contracts = {'caller': contract('caller', dependencies={'leaf': abi('i64')}), 'leaf': contract('leaf')}
        outgoing = {'caller': ['leaf'], 'leaf': []}
        self.assertEqual(contradictions('leaf', {'leaf': abi()}, contracts, outgoing)[0]['kind'], 'caller_declaration_conflict')
        self.assertEqual(contradictions('caller', {'caller': abi(), 'leaf': abi('i64')}, contracts, outgoing)[0]['kind'], 'callee_declaration_conflict')

    def test_worklist_revisits_caller_after_callee_change(self):
        state = {}
        calls = []
        def propose(name, current, issues):
            calls.append(name)
            if name == 'caller':
                return contract(name, dependencies={'leaf': current.get('leaf', {}).get('abi', abi('i64'))})
            return contract(name)
        report = reconcile_worklist(['caller', 'leaf'], state, {'caller': ['leaf'], 'leaf': []}, propose)
        self.assertGreaterEqual(calls.count('caller'), 2)
        self.assertTrue(report['declaration_consistent'])
        self.assertFalse(report['abi_proven'])

    def test_inconsistent_stable_hypotheses_are_not_convergence(self):
        def propose(name, current, issues):
            return contract(name, dependencies={'leaf': abi('i64')} if name == 'caller' else {})
        report = reconcile_worklist(['leaf', 'caller'], {}, {'caller': ['leaf'], 'leaf': []}, propose)
        self.assertFalse(report['declaration_consistent'])
        self.assertIn('caller', report['unresolved'])

    def test_recursive_group_terminates(self):
        def propose(name, current, issues):
            other = 'b' if name == 'a' else 'a'
            return contract(name, dependencies={other: abi()})
        state = {}
        report = reconcile_worklist(['a', 'b'], state, {'a': ['b'], 'b': ['a']}, propose, max_visits=3)
        self.assertTrue(report['declaration_consistent'])
        self.assertLessEqual(sum(report['visits'].values()), 6)

    def test_changed_contract_invalidates_transitive_callers(self):
        self.assertEqual(affected_callers('leaf', {'root': ['middle'], 'middle': ['leaf'], 'leaf': [], 'other': []}),
                         {'leaf', 'middle', 'root'})

    def test_baseline_packet_has_no_analysis_fields(self):
        f = {'name': 'f', 'symbolized_disassembly': '10: mov rax,[rip] # 20 <flag>\n ret',
             'calls': ['forbidden'], 'cfg': ['forbidden'], 'signature': 'forbidden'}
        plan = {'binary_data': {'objects': [{'name': 'flag', 'address': 32}],
                               'referenced_strings': [{'address': 32, 'text': 'yes'}, {'address': 48, 'text': 'no'}]},
                'runtime': {'samples': ['forbidden']}}
        packet = raw_evidence(f, plan, [])
        assert_baseline_boundary(packet)
        self.assertNotIn('forbidden', json.dumps(packet))
        self.assertEqual(len(packet['raw_elf_data']['strings']), 1)
        with self.assertRaisesRegex(ValueError, 'leaked'):
            assert_baseline_boundary({**packet, 'caller_sites': []})

    def test_packet_caps_auxiliary_evidence_without_truncating_assembly(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(range(len(messages[0]['content'])))
        config = {'tokenizer_path': '/fake', 'context_window': 260000, 'max_tokens': 260000,
                  'input_ceiling': 300, 'max_tokens_policy': 'remaining_context'}
        packet = {'assembly': 'ret', 'retrieved_examples': [{'llvm_ir': 'x' * 1000}], 'caller_sites': ['y' * 1000]}
        with patch('models.binary_decompilation.prompt_budget.tokenizer', return_value=FakeTokenizer()):
            packed, info = pack(config, 'translate', packet)
        self.assertEqual(packed['assembly'], 'ret')
        self.assertLessEqual(info['prompt_tokens'], 300)
        self.assertTrue(info['omitted'])

    def test_stage_budget_shared_across_new_contexts(self):
        config = {'model': 'fake', 'arm': 'baseline', 'max_attempts': 2, 'num_generate': 8}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(2):
                write_json(root / f'round_{i:03d}' / 'attempt_0/request.json', {'n': 8})
            with patch('models.binary_decompilation.reconciled_campaign.pack', return_value=({'assembly': 'ret'}, {})):
                result = batch(None, config, None, {'name': 'f'}, 'signature', root, {})
        self.assertEqual(result['requests'], 2)
        self.assertFalse(result['accepted'])

    def test_uncommitted_result_replayed_without_inference(self):
        from models.binary_decompilation.binary_project import digest
        from models.binary_decompilation.project_pipeline import generation_config
        config = {'model': 'fake', 'arm': 'baseline', 'max_attempts': 2, 'num_generate': 8}
        packet = {'assembly': 'ret'}
        from models.binary_decompilation.reconciliation_evidence import SIGNATURE
        identity = digest({'stage': 'signature', 'packet': packet, 'signature': '', 'settings': generation_config(config, 'signature')})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / 'round_000'
            write_json(d / 'input_identity.json', {'identity': identity})
            write_json(d / 'attempt_0/request.json', {'n': 8})
            write_json(d / 'result.json', {'accepted': True, 'contract': {}})
            with patch('models.binary_decompilation.reconciled_campaign.pack', return_value=(packet, {})), \
                 patch('shutil.disk_usage') as disk:
                disk.return_value.free = 100 * 1024**3
                result = batch(None, config, None, {'name': 'f'}, 'signature', root, {})
            self.assertTrue(result['accepted'])
            commit(result)
            self.assertTrue((d / 'committed.json').exists())


class NativeABIReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.reader = ABIReader(CLANG, cls.root / 'reader')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_compiler_derived_contract_abi(self):
        functions = self.reader.contract('f', 'extern int f(char*, unsigned long); extern void g(void);',
                                         'int f(char*, unsigned long);', ['f', 'g'], self.root / 'contract')
        self.assertEqual([p['type'] for p in functions['f']['parameters']], ['ptr', 'i64'])
        self.assertEqual(functions['g']['return_type'], 'void')

    def test_llvm_reader_preserves_variadic_and_abi_attributes(self):
        path = self.root / 'attributes.ll'
        path.write_text('declare zeroext i1 @f(i8 signext, ...)\n')
        result = self.reader.read(path)['f']
        self.assertTrue(result['variadic'])
        self.assertEqual(result['return_attributes'], ['zeroext'])
        self.assertEqual(result['parameters'][0]['attributes'], ['signext'])

    def test_call_sites_include_return_use(self):
        source, binary = self.root / 'calls.c', self.root / 'calls'
        source.write_text('__attribute__((noinline)) int leaf(int x) { return x+1; }\n'
                          'int main(int argc,char **argv) { return leaf(argc)==3; }\n')
        result = run([CLANG, '-O1', str(source), '-o', str(binary)])
        self.assertEqual(result['returncode'], 0)
        sites = extract_call_sites(binary, [{'name': 'leaf'}, {'name': 'main'}])
        self.assertEqual(sites['outgoing']['main'], ['leaf'])
        self.assertTrue(sites['incoming']['leaf'][0]['after'])

    def test_ir_abi_conflict_is_repairable_before_runtime(self):
        config = {'source_dir': str(self.root), 'build_dir': str(self.root), 'clang': CLANG,
                  'programs': {'fixture': {'tests': [], 'cases': []}}, 'signature_reconciliation': True}
        adapter = ReconciliationAdapter(config, 'fixture', self.root, self.reader, ['leaf'])
        adapter.contracts = {'leaf': contract('leaf')}
        candidate = self.root / 'wrong.ll'
        candidate.write_text('define i64 @leaf(i32 %x) { ret i64 0 }\n')
        result = adapter.validate('leaf', 'unused.c', 'int leaf(int);', candidate, self.root / 'validation', 'ir')
        self.assertEqual(result['failure_kind'], 'abi_conflict')
        self.assertTrue(result['repair_contract'])

    def test_call_site_type_cannot_hide_behind_correct_declaration(self):
        config = {'source_dir': str(self.root), 'build_dir': str(self.root), 'clang': CLANG,
                  'programs': {'fixture': {'tests': [], 'cases': []}}, 'signature_reconciliation': True}
        adapter = ReconciliationAdapter(config, 'fixture', self.root, self.reader, ['caller', 'leaf'])
        caller = contract('caller')
        caller['abi'] = abi(parameters=('ptr',))
        adapter.contracts = {'caller': caller, 'leaf': contract('leaf')}
        path = self.root / 'bad_call.ll'
        path.write_text('declare i32 @leaf(i32)\ndefine i32 @caller(ptr %x) {\n'
                        '%r = call i32 @leaf(ptr %x)\nret i32 %r\n}\n')
        result = adapter.validate('caller', 'unused.c', 'int caller(void *);', path, self.root / 'bad_call_validation', 'ir')
        self.assertEqual(result['failure_kind'], 'abi_conflict')
        self.assertIn('call_site', result['feedback'])

    def test_end_to_end_body_failure_revises_signature_and_executes_replacement(self):
        from models.binary_decompilation.reconciled_campaign import run_program
        from models.binary_decompilation.native_validation import elf_symbols
        root = self.root / 'end_to_end'
        (root / 'src').mkdir(parents=True, exist_ok=True)
        source, binary = root / 'src/fixture.c', root / 'src/fixture'
        source.write_text('#include <stdio.h>\n#include <stdlib.h>\n'
                          '__attribute__((noinline)) int leaf(int x) { return x+7; }\n'
                          'int main(int argc,char **argv) { printf("%d\\n",leaf(atoi(argv[1]))); }\n')
        built = run([CLANG, '-O1', '-fPIE', '-pie', str(source), '-o', str(binary)])
        self.assertEqual(built['returncode'], 0, built)
        entry = hex(elf_symbols(binary)['leaf']['address'])
        function = {'name': 'leaf', 'entry': entry, 'source_file': 'src/fixture.c',
                    'symbolized_disassembly': run(['objdump', '-d', '--disassemble=leaf', str(binary)])['stdout']}
        plan = {'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'targets': [function], 'components': [['leaf']], 'binary_data': {}, 'runtime': {}}
        write_json(root / 'plan_cache/retrieval/fixture/plan.json', plan)
        client = Mock()
        prompts = []
        def create(**kwargs):
            prompt = kwargs['messages'][0]['content']
            prompts.append(prompt)
            if prompt.startswith('Infer'):
                evidence = json.loads(prompt.split('\nEvidence:\n', 1)[1])
                declaration = 'int leaf(int);' if 'previous_contract' in evidence else 'long leaf(int);'
                text = json.dumps({'c_declaration': declaration, 'project_header': declaration, 'direct_probes': []})
            else:
                text = '```llvm\ndefine i32 @leaf(i32 %x) { %y = add i32 %x, 7\n ret i32 %y\n}\n```'
            response = Mock()
            response.model_dump.return_value = {'choices': [{'index': i, 'finish_reason': 'stop', 'message': {'content': text}} for i in range(8)]}
            return response
        client.chat.completions.create.side_effect = create
        config = {'model': 'fixture', 'arm': 'reconciled', 'signature_reconciliation': True,
                  'source_dir': str(root), 'build_dir': str(root), 'clang': CLANG,
                  'programs': {'fixture': {'tests': [], 'cases': [{'args': ['3']}, {'args': ['-2']}]}},
                  'plan_cache': str(root / 'plan_cache'), 'output_dir': str(root / 'out'),
                  'retrieval_cache': str(root / 'retrieval'), 'retrieval': {'enabled': False},
                  'context_window': 260000, 'max_tokens': 1024, 'max_attempts': 3, 'num_generate': 8}
        config['min_free_output_gib'] = 0
        (root / 'out').mkdir()
        with patch('models.binary_decompilation.reconciled_campaign.model_client', return_value=client), \
             patch('models.binary_decompilation.service_preflight.check_services', return_value={'ready': True}), \
             patch('models.binary_decompilation.reconciled_campaign.pack', side_effect=lambda c,i,e: (e, {})):
            summary = run_program(config, 'fixture')
        self.assertEqual(summary['accepted_ir'], 1, summary)
        state = json.loads((root / 'out/retrieval/fixture/state.json').read_text())
        self.assertEqual(state['contracts']['leaf']['abi']['return_type'], 'i32')
        self.assertGreaterEqual(state['reviews']['leaf'], 1)
        self.assertTrue(state['ir']['leaf']['validation']['execution_observed'])
        self.assertTrue(any('body_validation_review' in p for p in prompts))
        prompts.clear()
        config.update(arm='baseline', signature_reconciliation=False, output_dir=str(root / 'baseline'))
        (root / 'baseline').mkdir()
        with patch('models.binary_decompilation.reconciled_campaign.model_client', return_value=client), \
             patch('models.binary_decompilation.service_preflight.check_services', return_value={'ready': True}), \
             patch('models.binary_decompilation.reconciled_campaign.pack', side_effect=lambda c,i,e: (e, {})), \
             patch('models.binary_decompilation.reconciled_campaign.analyze_function', side_effect=AssertionError('Baseline must not analyze')):
            baseline = run_program(config, 'fixture')
        self.assertEqual(baseline['accepted_ir'], 0)
        self.assertFalse(any('body_validation_review' in p for p in prompts))
        for prompt in prompts:
            assert_baseline_boundary(json.loads(prompt.split('\nEvidence:\n', 1)[1]))


if __name__ == '__main__':
    unittest.main()
