"""Persistent experiment supervisor with isolated logs, lifecycle and comparison."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from models.binary_decompilation.binary_project import load_config
from models.binary_decompilation.project_pipeline import write_json
from models.binary_decompilation.project_report import report


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def compare(baseline, output):
    rows = []
    for root, label in ((baseline, 'baseline'), (output, 'guided')):
        for path in sorted(root.glob('retrieval/*/summary.json')):
            summary = json.loads(path.read_text())
            rows.append({'run': label, 'program': path.parent.name,
                         **{k: summary.get(k) for k in ('targets', 'accepted_ir', 'accepted_c', 'complete_project_recovery')}})
    result = {'rows': rows, 'interpretation': 'Bundled workflow comparison, not a controlled model or component ablation.'}
    write_json(output / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config['output_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'phase': 'starting', 'runner_pid': os.getpid(), 'started_at': now()}
        command = [sys.executable, '-u', '-m', 'models.binary_decompilation.binary_project',
                   '--config', str(args.config.resolve()), '--track', 'retrieval', '--parallel-programs']
        status['command'] = command
        write_json(root / 'runner_status.json', status)
        with (root / 'campaign.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status.update(phase='running', campaign_pid=child.pid)
            write_json(root / 'runner_status.json', status)

            def stop(signum, frame):
                child.send_signal(signal.SIGINT)

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            status['campaign_exit_code'] = child.wait()
        try:
            report(root)
            compare(args.baseline, root)
            status['phase'] = 'finished' if status['campaign_exit_code'] == 0 else 'process_failed'
        except Exception as error:
            status.update(phase='report_failed', error=str(error))
            raise
        finally:
            status['finished_at'] = now()
            status['note'] = 'Process completion is not successful standalone recovery.'
            write_json(root / 'runner_status.json', status)
    return status['campaign_exit_code']


if __name__ == '__main__':
    sys.exit(main())
