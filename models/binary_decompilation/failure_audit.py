"""Reproducible artifact audit; terminal failures are not all model failures."""

import argparse
from collections import Counter
import json
from pathlib import Path

from models.binary_decompilation.project_pipeline import write_json


def audit(root):
    rows, counts, finishes, terminals = [], Counter(), Counter(), Counter()
    for results_path in sorted(root.glob('retrieval/*/results.json')):
        program = results_path.parent.name
        for result in json.loads(results_path.read_text()):
            accepted = bool(result.get('ir', {}).get('accepted') and result.get('c', {}).get('accepted'))
            error = result.get('error') or result.get('signature', {}).get('error')
            if accepted:
                category = 'accepted'
            elif error and 'timed out' in error.lower():
                category = 'terminal_timeout'
            elif result.get('ir', {}).get('needs_probe'):
                category = 'missing_execution_and_probe_failure'
            else:
                category = 'generation_or_validation_failure'
            terminals[category] += 1
            stages = {s: {k: result.get(s, {}).get(k) for k in ('accepted', 'attempts', 'reason', 'error')}
                      for s in ('signature', 'ir', 'probes', 'c')}
            rows.append({'program': program, 'function': result['name'], 'category': category,
                         'error': error, 'stages': stages,
                         'artifact': str(results_path.parent / 'functions' / result['name'] / 'result.json')})
    for path in root.glob('retrieval/*/functions/*/*/*/attempt_*/selection.json'):
        stage = path.parents[2].name
        for candidate in json.loads(path.read_text())['candidates']:
            counts[stage + ':' + candidate['validation'].get('failure_kind', 'unknown')] += 1
            finishes[stage + ':' + str(candidate.get('finish_reason'))] += 1
    return {'baseline': str(root.resolve()), 'terminal_categories': dict(terminals), 'functions': rows,
            'all_saved_revision_candidate_outcomes': dict(counts), 'finish_reasons': dict(finishes),
            'limitations': ['Candidate counts include historical pilot/restart revisions, unlike current terminal summaries',
                            'A terminal timeout does not establish the earlier candidates were semantically correct',
                            'Accepted replacement tests are not standalone recovery or equivalence proofs']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.output, result)
    print(json.dumps(result['terminal_categories']))


if __name__ == '__main__':
    main()
