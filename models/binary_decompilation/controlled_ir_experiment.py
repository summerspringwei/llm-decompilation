"""Frozen, resumable screening and paired IR-only Coreutils experiments."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from models.binary_decompilation.binary_project import load_config, BinaryCoreutilsAdapter
from models.binary_decompilation.native_validation import make_wrapper
from models.binary_decompilation.project_pipeline import write_json
from models.binary_decompilation.validation_feedback import runtime_feedback


SCREEN = {'basename': ['oputs_', 'perform_basename', 'main'],
          'uniq': ['find_field', 'writeline', 'main'], 'df': ['find_mount_point', 'get_dev']}
ARMS = {'A': {}, 'B': {'semantic_analysis': True},
        'C': {'semantic_analysis': True, 'translation_scaffold': True},
        'D': {'semantic_analysis': True, 'translation_scaffold': True, 'semantic_checks': True}}
SEEDS = (1729, 1730)
HELDOUT = {
    'basename': [{'args': ['--', '']}, {'args': ['dir/a.tar.gz', '.gz']},
                 {'args': ['-a', 'path with spaces/a', './b/', 'c']},
                 {'args': ['-z', '-s', '.txt', 'a.txt', 'b.txt']},
                 {'args': ['name', 'name']}, {'args': ['//a///']}, {'args': ['--help', '--version']}],
    'uniq': [{'args': ['-c'], 'stdin': 'x\nx\nx\ny\nz\nz\n'},
             {'args': ['-d', '-i'], 'stdin': 'Ab\naB\nCD\n'},
             {'args': ['-f', '1'], 'stdin': '1 a\n2 a\n3 b\n'},
             {'args': ['-s', '2', '-w', '1'], 'stdin': '00ax\n11ay\n22bz\n'},
             {'args': ['-z', '-c'], 'stdin': 'a\u0000a\u0000b\u0000'},
             {'args': ['--all-repeated=separate'], 'stdin': 'a\na\nb\nc\nc\nc\n'},
             {'args': ['-u'], 'stdin': 'a\na\nb'}, {'args': [], 'stdin': '\n\n\nx\nx\n'},
             {'args': ['--help', '--version']}],
    'df': [{'args': ['--output=source,size,target', '.']}, {'args': ['-B', '512', '--output=source,size', '/']},
           {'args': ['--output=source,fstype,target', '.']},
           {'args': ['--output=itotal', '/']},
           {'args': ['--output=source,invalid_probe_field']}, {'args': ['/__decomp_missing_path_9e31__']},
           {'args': ['--help', '--version']}],
}


def prepare(base_path, output):
    output = Path(output).resolve()
    implementations = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.glob('*.py')}
    identity_path = output / 'implementation.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != implementations:
        raise ValueError('Framework changed after protocol freeze; use a new output root')
    write_json(identity_path, implementations)
    base = load_config(Path(base_path))
    base.pop('extends', None)
    base.update(ir_only=True, recover_standalone=False, guided_decompilation=False, compiler_analysis=True,
                semantic_analysis=False, translation_scaffold=False, semantic_checks=False,
                enable_thinking=True, stage_enable_thinking={'signature': False, 'probes': False},
                num_generate=8, max_attempts=10, context_window=260000, max_tokens=260000,
                max_tokens_policy='remaining_context', stage_max_tokens={},
                request_timeout=600, service_health_timeout=60,
                retrieval_cache=str(output / 'retrieval_cache'))
    base['retrieval'].update(candidate_pool=20, structural_rerank=True, max_examples=2)
    configs = {}
    for phase in ('screen', 'full'):
        for arm, flags in ARMS.items():
            for seed in SEEDS[:1] if phase == 'screen' else SEEDS:
                label = f'{phase}_{arm}_{seed}'
                config = copy.deepcopy(base)
                config.update(flags, seed=seed, output_dir=str(output / label))
                config.pop('function_filter', None)
                config['function_filters'] = SCREEN if phase == 'screen' else {}
                config['experiment_arm'] = arm
                path = output / 'configs' / (label + '.json')
                if path.exists() and json.loads(path.read_text()) != config:
                    raise ValueError('Frozen experiment configuration changed: ' + str(path))
                write_json(path, config)
                configs[label] = str(path)
    protocol = {'schema_version': 1, 'arms': ARMS, 'screen': SCREEN, 'seeds': SEEDS,
                'configs': configs, 'heldout': HELDOUT,
                'selection': 'max development IR accepted, then min measured tokens, then simpler arm',
                'scope': 'IR function replacement, not standalone project recovery',
                'semantic_checker_domain': 'scalar call-free integer functions; other functions are unknown',
                'inference_source_policy': 'binary/public ABI only; no application source bodies or reference IR'}
    frozen = output / 'protocol.json'
    serializable = json.loads(json.dumps(protocol))
    if frozen.exists() and json.loads(frozen.read_text()) != serializable:
        raise ValueError('Protocol is immutable; use a new output root')
    write_json(frozen, protocol)
    return protocol


def summarize(root):
    results, summaries = [], []
    for program in SCREEN:
        directory = root / 'retrieval' / program
        if not (directory / 'summary.json').exists():
            raise RuntimeError('Incomplete campaign: ' + str(directory))
        summary = json.loads((directory / 'summary.json').read_text())
        expected = set(summary['selected_functions'])
        rows = json.loads((directory / 'results.json').read_text())
        if {r['name'] for r in rows} != expected:
            raise RuntimeError('Missing selected function outcomes: ' + str(directory))
        results.extend(rows)
        summaries.append(summary)
    tokens, usage_missing = 0, 0
    for response in root.glob('retrieval/*/functions/*/*/*/attempt_*/response.json'):
        usage = json.loads(response.read_text()).get('usage')
        if usage and usage.get('total_tokens') is not None:
            tokens += usage['total_tokens']
        else:
            usage_missing += 1
    return {'evaluated': len(results), 'accepted_ir': sum(bool(r.get('ir', {}).get('accepted')) for r in results),
            'total_tokens': tokens, 'responses_without_usage': usage_missing,
            'rows': [{'name': r['name'], 'accepted_ir': bool(r.get('ir', {}).get('accepted'))} for r in results]}


def select_arm(screens):
    return max(('B', 'C', 'D'), key=lambda arm: (
        screens[arm]['accepted_ir'],
        -screens[arm]['total_tokens'] if not screens[arm]['responses_without_usage'] else -float('inf'),
        -len(ARMS[arm])))


def evaluate_heldout(config_path, output):
    config = load_config(Path(config_path))
    rows = []
    for program in SCREEN:
        directory = Path(config['output_dir']) / 'retrieval' / program
        outcomes = json.loads((directory / 'results.json').read_text())
        for result in outcomes:
            row = {'program': program, 'function': result['name'], 'accepted': False}
            recovered = result.get('ir', {})
            if not recovered.get('accepted'):
                row['reason'] = 'Not accepted on development tests'
                rows.append(row)
                continue
            manifest = recovered['validation'].get('manifest')
            if not manifest:
                raise RuntimeError('Accepted IR is missing its validation manifest')
            target = Path(output) / program / result['name']
            frozen_config = copy.deepcopy(config)
            frozen_config['programs'][program]['tests'] = []
            frozen_config['programs'][program]['cases'] = HELDOUT[program]
            adapter = BinaryCoreutilsAdapter(frozen_config, program, target)
            wrapper = make_wrapper(adapter.clang, adapter.binary, [], [manifest], target / 'runtime')
            tests = adapter.feedback_test(wrapper, target / 'tests')
            events_path = target / 'runtime/events.jsonl'
            events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
            diagnostic, _ = runtime_feedback(tests, events, result['name'])
            exercised = diagnostic['coverage']['target_exercising_cases']
            row.update(accepted=tests['failed'] == 0 and exercised > 0, tests=tests, diagnostic=diagnostic,
                       reason='Differential held-out execution' if exercised else 'Target not exercised by held-out cases')
            write_json(target / 'result.json', row)
            rows.append(row)
    result = {'rows': rows, 'accepted_ir': sum(r['accepted'] for r in rows), 'targets': len(rows),
              'note': 'Held-out evidence is never fed back into generation; missing coverage is not acceptance'}
    write_json(Path(output) / 'summary.json', result)
    return result


def run_campaign(protocol, label, root):
    config_path = Path(protocol['configs'][label])
    config = load_config(config_path)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    finished = output / 'controlled_completion.json'
    command = [sys.executable, '-u', '-m', 'models.binary_decompilation.binary_project', '--config', str(config_path),
               '--track', 'retrieval', '--parallel-programs']
    if not finished.exists():
        with (output / 'supervisor.log').open('a') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            previous = {}
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, lambda signum, frame: child.send_signal(signal.SIGINT))
            write_json(root / 'runner_status.json', {'phase': label, 'pid': os.getpid(), 'child': child.pid,
                       'started_unix': time.time(), 'command': command})
            try:
                code = child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            if code:
                raise RuntimeError(label + ' failed with process exit ' + str(code))
        summary = summarize(output)
        write_json(finished, summary)
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'coreutils_binary_candidates8.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Otherwise freeze configurations without inference')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(args.config, args.output)
        if not args.run:
            print(json.dumps({'prepared': str(args.output / 'protocol.json'), 'inference_started': False}))
            return
        try:
            screens = {arm: run_campaign(protocol, f'screen_{arm}_{SEEDS[0]}', args.output) for arm in ARMS}
            selected = select_arm(screens)
            write_json(args.output / 'selection.json', {'selected_arm': selected, 'development_only': True, 'screens': screens})
            full = {}
            for seed in SEEDS:
                for arm in ('A', selected):
                    label = f'full_{arm}_{seed}'
                    full[label] = run_campaign(protocol, label, args.output)
            heldout = {label: evaluate_heldout(protocol['configs'][label], args.output / 'heldout' / label) for label in full}
            improvements = [heldout[f'full_{selected}_{seed}']['accepted_ir'] > heldout[f'full_A_{seed}']['accepted_ir'] for seed in SEEDS]
            write_json(args.output / 'comparison.json', {'screens': screens, 'selected': selected, 'full': full,
                       'heldout': heldout, 'improved_on_both_seeds': all(improvements)})
            write_json(args.output / 'runner_status.json', {'phase': 'finished', 'pid': os.getpid(),
                       'improved_on_both_seeds': all(improvements), 'finished_unix': time.time()})
        except BaseException as error:
            write_json(args.output / 'runner_status.json', {'phase': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       'error': str(error), 'pid': os.getpid(), 'finished_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
