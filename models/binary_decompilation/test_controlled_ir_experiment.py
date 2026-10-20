"""Protocol immutability, selection isolation, seeds, and IR-only settings."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from models.binary_decompilation.controlled_ir_experiment import prepare, select_arm, SCREEN, HELDOUT
from models.binary_decompilation.project_pipeline import generate, generation_metadata


class ControlledIRTests(unittest.TestCase):
    def test_frozen_protocol_and_matched_arms(self):
        base = Path(__file__).with_name('configs') / 'coreutils_binary_candidates8.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = prepare(base, root)
            self.assertEqual(len(protocol['configs']), 12)
            configs = [json.loads(Path(protocol['configs'][f'screen_{a}_1729']).read_text()) for a in 'ABCD']
            for config in configs:
                self.assertTrue(config['ir_only'])
                self.assertTrue(config['enable_thinking'])
                self.assertEqual(config['num_generate'], 8)
                self.assertEqual(config['max_attempts'], 10)
                self.assertEqual(config['function_filters'], SCREEN)
                self.assertEqual(config['retrieval'], configs[0]['retrieval'])
                self.assertEqual(config['programs'], configs[0]['programs'])
                self.assertNotIn('heldout', config)
            self.assertEqual(prepare(base, root), protocol)
            path = Path(protocol['configs']['screen_A_1729'])
            changed = json.loads(path.read_text())
            changed['seed'] = 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                prepare(base, root)

    def test_selection_uses_development_only_and_simplicity(self):
        rows = {a: {'accepted_ir': 3, 'total_tokens': 100, 'responses_without_usage': 0} for a in 'ABCD'}
        self.assertEqual(select_arm(rows), 'B')
        rows['C']['total_tokens'] = 80
        self.assertEqual(select_arm(rows), 'C')
        rows['D']['accepted_ir'] = 4
        self.assertEqual(select_arm(rows), 'D')

    def test_seed_reaches_api(self):
        client = Mock()
        client.chat.completions.create.return_value.model_dump.return_value = {'choices': []}
        config = {'model': 'fixture', 'seed': 1729}
        generate(client, config, 'fixture', Path('/unused'))
        self.assertEqual(client.chat.completions.create.call_args.kwargs['seed'], 1729)
        self.assertEqual(generation_metadata(config)['seed'], 1729)

    def test_heldout_cases_are_distinct_from_development(self):
        from models.binary_decompilation.binary_project import load_config
        config = load_config(Path(__file__).with_name('configs') / 'coreutils_binary_candidates8.json')
        for program, cases in HELDOUT.items():
            for case in cases:
                self.assertNotIn(case, config['programs'][program]['cases'])


if __name__ == '__main__':
    unittest.main()
