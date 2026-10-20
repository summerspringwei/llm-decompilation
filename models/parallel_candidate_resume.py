"""Resume a frozen reconciliation campaign with globally bounded DP requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from models.binary_decompilation import project_pipeline as pipeline
from models.binary_decompilation import reconciled_campaign as campaign

ORIGINAL_GENERATE = pipeline.generate


def parallel_generate(client, config, prompt, directory):
    lock_root = Path(os.environ['DECOMP_DP_LOCK_DIR'])
    lock_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    transport_failed = threading.Event()

    def candidate(index):
        root = directory / f'dp_candidate_{index}'
        root.mkdir(exist_ok=True)
        response = root / 'response.json'
        if response.exists():
            return json.loads(response.read_text())
        if (root / 'request.json').exists():
            return {'choices': [], 'request_error': {'type': 'InterruptedRequest'}}
        settings = dict(config, num_generate=1)
        if config.get('seed') is not None:
            settings['seed'] = (config['seed'] + index) % (2**31)
        slot = None
        while slot is None:
            if transport_failed.is_set():
                return {'choices': [], 'request_error': {'type': 'NotSubmittedAfterTransportFailure'}}
            for number in range(3):
                handle = (lock_root / f'slot_{number}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    slot = handle
                    break
                except BlockingIOError:
                    handle.close()
            if slot is None:
                time.sleep(0.2)
        try:
            pipeline.write_json(root / 'request.json', {
                'n': 1, 'seed': settings.get('seed'), 'started_unix': time.time(),
                'candidate_index': index, 'global_concurrency': 3})
            try:
                result = ORIGINAL_GENERATE(client, settings, prompt, root)
            except Exception as error:
                result = {'choices': [], 'request_error': {
                    'type': type(error).__name__, 'message': str(error)}}
            pipeline.write_json(response, result)
            if result.get('request_error'):
                transport_failed.set()
            return result
        finally:
            slot.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(candidate, range(config.get('num_generate', 1))))
    errors = [r.get('request_error') for r in results if r.get('request_error')]
    if errors:
        pipeline.write_json(directory / 'transport_failure.json', {
            'errors': errors, 'budget_policy': 'Infrastructure interruption, not model correctness failure',
            'completed_candidates_preserved': True})
        raise RuntimeError('Inference transport failed; stopping campaign before further generation rounds')
    choices = []
    for index, result in enumerate(results):
        choice = dict((result.get('choices') or [{}])[0])
        choice.update(index=index)
        if not choice.get('message'):
            choice.update(message={'content': ''}, finish_reason='interrupted')
        choices.append(choice)
    complete_usage = all(r.get('usage') and all(
        r['usage'].get(k) is not None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        for r in results)
    usage = ({k: sum(r['usage'][k] for r in results)
              for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
             if complete_usage else None)
    return {'choices': choices, 'usage': usage,
            'elapsed_seconds': time.monotonic() - started,
            'candidate_errors': [r.get('request_error') for r in results],
            'transport': 'eight independent n=1 requests, global concurrency three'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--program')
    args = parser.parse_args()
    root = args.output.resolve()
    os.environ['DECOMP_DP_LOCK_DIR'] = str(root / 'dp_slots')
    pipeline.generate = parallel_generate
    if args.worker_config:
        campaign.run_program(campaign.load_config(args.worker_config), args.program)
        return
    configs = campaign.prepare(root, Path(campaign.__file__).with_name('configs') / 'coreutils_binary_candidates8.json')
    with (root / 'dp_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        def stop(signum, frame):
            raise KeyboardInterrupt('Signal ' + str(signum))
        signal.signal(signal.SIGTERM, stop)
        try:
            for arm, config_path in configs.items():
                config = campaign.load_config(config_path)
                children = []
                for program in config['programs']:
                    if (Path(config['output_dir']) / 'retrieval' / program / 'summary.json').exists():
                        continue
                    log = root / 'logs' / f'dp_{arm}_{program}.log'
                    log.parent.mkdir(exist_ok=True)
                    with log.open('a') as stream:
                        children.append(subprocess.Popen([
                            sys.executable, '-u', '-m', 'models.parallel_candidate_resume',
                            '--output', str(root), '--worker-config', str(config_path), '--program', program],
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
                pipeline.write_json(root / 'runner_status.json', {
                    'phase': arm, 'pid': os.getpid(), 'transport': 'dp_individual_candidates',
                    'global_concurrency': 3, 'workers': [p.pid for p in children]})
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None, 0) for p in children):
                        raise RuntimeError('DP worker failed; see logs')
                    time.sleep(2)
                if any(p.returncode for p in children):
                    raise RuntimeError('DP worker failed; see logs')
            # All generation summaries exist; the original supervisor runs held-out tests and reporting.
            campaign.supervise(root, configs)
        except BaseException as error:
            pipeline.write_json(root / 'runner_status.json', {
                'phase': 'stopped', 'pid': os.getpid(), 'error': str(error)})
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                child.wait()


if __name__ == '__main__':
    main()
