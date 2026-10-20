import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from models import parallel_candidate_resume as dp


class ParallelCandidateTest(unittest.TestCase):
    def test_independent_candidates_budget_order_and_cache(self):
        active = peak = 0
        seeds = []
        guard = threading.Lock()

        def generate(client, config, prompt, root):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
                seeds.append(config['seed'])
            self.assertEqual(config['num_generate'], 1)
            time.sleep(0.02)
            with guard:
                active -= 1
            return {'choices': [{'index': 0, 'message': {'content': str(config['seed'])},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(dp, 'ORIGINAL_GENERATE', generate), \
                patch.dict(os.environ, {'DECOMP_DP_LOCK_DIR': tmp + '/locks'}):
            result = dp.parallel_generate(None, {'num_generate': 8, 'seed': 1729}, '', Path(tmp))
            self.assertEqual(sorted(seeds), list(range(1729, 1737)))
            self.assertEqual([c['index'] for c in result['choices']], list(range(8)))
            self.assertEqual(result['usage']['total_tokens'], 40)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 3)
            dp.parallel_generate(None, {'num_generate': 8, 'seed': 1729}, '', Path(tmp))
            self.assertEqual(len(seeds), 8)

    def test_transport_failure_preserves_siblings_and_missing_usage(self):
        def generate(client, config, prompt, root):
            if config['seed'] == 2:
                raise TimeoutError('fixture')
            return {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(dp, 'ORIGINAL_GENERATE', generate), \
                patch.dict(os.environ, {'DECOMP_DP_LOCK_DIR': tmp + '/locks'}):
            with self.assertRaisesRegex(RuntimeError, 'Inference transport failed'):
                dp.parallel_generate(None, {'num_generate': 3, 'seed': 1}, '', Path(tmp))
            self.assertTrue((Path(tmp) / 'transport_failure.json').exists())
            self.assertTrue((Path(tmp) / 'dp_candidate_0/response.json').exists())
            self.assertEqual(json.loads((Path(tmp) / 'dp_candidate_1/response.json').read_text())[
                'request_error']['type'], 'TimeoutError')


if __name__ == '__main__':
    unittest.main()
