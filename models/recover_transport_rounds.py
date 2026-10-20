"""Archive all-transport-failed rounds without erasing their audit records."""

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def recover(root):
    archive = root / 'recovery_archive' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rounds, affected = [], set()
    for arm in ('baseline', 'reconciled'):
        for response in (root / arm).glob('retrieval/*/functions/*/*/round_*/attempt_0/response.json'):
            request = json.loads((response.parent / 'request.json').read_text())
            children = list(response.parent.glob('dp_candidate_*/response.json'))
            if len(children) != request['n']:
                continue
            results = [json.loads(p.read_text()) for p in children]
            if not all(r.get('request_error') for r in results):
                continue
            if any(c.get('finish_reason') == 'stop' for r in results for c in r.get('choices', [])):
                continue
            program = response.parents[5]
            state = json.loads((program / 'state.json').read_text())
            name = response.parents[3].name
            if state.get('ir', {}).get(name, {}).get('accepted'):
                continue
            rounds.append(response.parents[1])
            affected.add(program)
    if not rounds:
        return {'restored_rounds': 0}
    archive.mkdir(parents=True)

    def move(path):
        if path.exists():
            target = archive / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))

    for program in affected:
        target = archive / (program / 'state.json').relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(program / 'state.json', target)
        for name in ('summary.json', 'results.json'):
            move(program / name)
    for path in rounds:
        move(path)
    for name in ('comparison.json', 'heldout', 'runner_status.json'):
        move(root / name)
    record = {'archive': str(archive), 'restored_rounds': len(rounds),
              'rounds': [str(p.relative_to(root)) for p in rounds],
              'affected_programs': [str(p.relative_to(root)) for p in sorted(affected)],
              'policy': 'Only entirely transport-failed batches restored; mixed and genuine model attempts retained'}
    (archive / 'recovery.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = recover(args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != 'rounds'}))
