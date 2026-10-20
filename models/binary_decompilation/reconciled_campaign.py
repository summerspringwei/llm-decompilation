"""Isolated no-analysis baseline and bidirectional signature-reconciliation runner."""

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

from models.binary_decompilation.binary_project import load_config, digest
from models.binary_decompilation.candidate_search import search_candidates
from models.binary_decompilation.controlled_ir_experiment import evaluate_heldout
from models.binary_decompilation.project_pipeline import (
    Retriever, generation_config, model_client, run, write_json,
)
from models.binary_decompilation.reconciliation_evidence import (
    SIGNATURE, IR, PROBES, raw_evidence, compact_analysis, pack, assert_baseline_boundary,
)
from models.binary_decompilation.semantic_analysis import analyze_function
from models.binary_decompilation.signature_reconciliation import (
    ABIReader, ReconciliationAdapter, extract_call_sites, reconcile_worklist,
    contradictions, compact_contract, contract_fingerprint,
)


def affected_callers(name, outgoing):
    affected, pending = {name}, [name]
    while pending:
        callee = pending.pop()
        for caller, deps in outgoing.items():
            if callee in deps and caller not in affected:
                affected.add(caller)
                pending.append(caller)
    return affected


def batch(client, config, adapter, function, stage, root, evidence, signature=''):
    root.mkdir(parents=True, exist_ok=True)
    settings = generation_config(config, stage)
    limit = min(config['max_attempts'], 3) if stage == 'probes' else config['max_attempts']
    used = len(list(root.glob('round_*/attempt_0/request.json')))
    instruction = {'signature': SIGNATURE, 'ir': IR, 'probes': PROBES}[stage]
    try:
        packet, packing = pack(settings, instruction, evidence)
    except ValueError as error:
        return {'accepted': False, 'failure_kind': 'context_window', 'feedback': str(error),
                'requests': used, 'terminal': True}
    if config['arm'] == 'baseline':
        assert_baseline_boundary(packet)
    identity = digest({'stage': stage, 'packet': packet, 'signature': signature, 'settings': settings})
    # An action saved before its caller checkpoint can be safely replayed without a new request.
    for old in sorted(root.glob('round_*')):
        metadata = old / 'input_identity.json'
        if metadata.exists() and json.loads(metadata.read_text()).get('identity') == identity and not (old / 'committed.json').exists():
            directory = old
            break
    else:
        if used >= limit:
            return {'accepted': False, 'reason': 'Shared stage generation budget exhausted', 'requests': used}
        index = max([int(p.name.split('_')[-1]) for p in root.glob('round_*')], default=-1) + 1
        directory = root / f'round_{index:03d}'
    if used >= limit and not (directory / 'attempt_0/request.json').exists():
        return {'accepted': False, 'reason': 'Shared stage generation budget exhausted', 'requests': used}
    for path, minimum in ((root, config.get('min_free_output_gib', 15) * 1024**3),
                          (Path('/tmp'), config.get('min_free_temp_gib', 2) * 1024**3)):
        import shutil
        if shutil.disk_usage(path).free < minimum:
            raise RuntimeError('Insufficient disk space; refusing new inference: ' + str(path))
    directory.mkdir(exist_ok=True)
    write_json(directory / 'input_identity.json', {'identity': identity})
    write_json(directory / 'evidence_packing.json', packing)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = search_candidates(client, {**settings, 'max_attempts': 1}, adapter, function, stage,
                                   directory, signature, instruction, packet)
    result['_batch_dir'] = str(directory)
    result['requests'] = len(list(root.glob('round_*/attempt_0/request.json')))
    if result.get('validation', {}).get('failure_kind') == 'infrastructure':
        raise RuntimeError(result['validation']['feedback'])
    return result


def commit(result):
    if result and result.get('_batch_dir'):
        write_json(Path(result['_batch_dir']) / 'committed.json', {'state': 'checkpointed'})


def attach_abi(result):
    if result.get('accepted'):
        validation = result['validation']
        result.update(abi=validation['abi'], dependencies=validation.get('dependencies', {}),
                      status='syntax_valid_abi_hypothesis', abi_verified=False)
    return result


def run_program(config, program):
    directory = Path(config['output_dir']) / 'retrieval' / program
    directory.mkdir(parents=True, exist_ok=True)
    cached = Path(config['plan_cache']) / 'retrieval' / program / 'plan.json'
    plan = json.loads(cached.read_text())
    actual_hash = hashlib.sha256(Path(plan['binary']).read_bytes()).hexdigest()
    if actual_hash != plan['binary_sha256']:
        raise ValueError('Cached inventory belongs to a different binary')
    write_json(directory / 'plan.json', plan)
    functions = {f['name']: f for f in plan['targets']}
    if config.get('function_filters', {}).get(program):
        selected = set(config['function_filters'][program])
        if selected - functions.keys():
            raise ValueError('Unknown selected functions')
        functions = {n: f for n, f in functions.items() if n in selected}
    reader = ABIReader(config['clang'], directory / 'abi_reader')
    adapter = ReconciliationAdapter(config, program, directory, reader, functions)
    original = adapter.feedback_test(adapter.binary, directory / 'original_tests')
    if original['failed'] or not original['passed']:
        raise RuntimeError('Original binary failed development tests')
    write_json(directory / 'original_tests.json', original)
    client = model_client(config)
    from models.binary_decompilation.service_preflight import check_services
    with (Path(config['output_dir']).parent / 'service_preflight.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        health = check_services(client, config)
    write_json(directory / 'service_preflight.json', health)
    if not health['ready']:
        raise RuntimeError('Service preflight failed')

    enhanced = config['arm'] == 'reconciled'
    ordered = ([n for component in plan['components'] for n in component if n in functions] if enhanced else
               sorted(functions, key=lambda n: int(functions[n]['entry'], 16)))
    write_json(directory / 'generation_order.json', {'functions': ordered,
               'policy': 'callee-first SCC order' if enhanced else 'binary address order; no call-graph scheduling'})
    retriever = Retriever(config['retrieval'])
    base_evidence, facts = {}, {}
    sites = extract_call_sites(adapter.binary, list(functions.values())) if enhanced else {'incoming': {}, 'outgoing': {}}
    outgoing = sites['outgoing']
    if enhanced:
        for name, function in functions.items():
            outgoing[name] = sorted(set(outgoing.get(name, [])) | {
                c['name'] for c in function.get('calls', []) if c['name'] in functions})
        sites['limitations'].append('Graph dependencies also include conservative address-taken/runtime edges from the inventory; these do not supply concrete call-site windows')
        write_json(directory / 'call_sites.json', sites)
    for name in ordered:
        f = functions[name]
        root = directory / 'functions' / name
        root.mkdir(parents=True, exist_ok=True)
        cache = Path(config['retrieval_cache']) / digest({'assembly': f['symbolized_disassembly'], 'retrieval': config['retrieval']}) / 'examples.json'
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not cache.exists():
                write_json(cache, retriever.retrieve(adapter.binary, name))
            examples = json.loads(cache.read_text())
        write_json(root / 'retrieval.json', examples)
        base_evidence[name] = raw_evidence(f, plan, examples)
        if enhanced:
            facts[name] = analyze_function(adapter.binary, f, root / 'analysis')
    state_path = directory / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'contracts': {}, 'ir': {}, 'reviews': {}, 'reconciliation': []}
    contracts = state['contracts']
    adapter.contracts, adapter.outgoing = contracts, outgoing

    def checkpoint():
        write_json(state_path, state)

    def evidence(name):
        packet = copy.deepcopy(base_evidence[name])
        if enhanced:
            packet.update(compact_analysis(facts[name], name, sites, contracts, plan['runtime'], outgoing))
        return packet

    def changed(name, old, new):
        for affected in affected_callers(name, outgoing):
            if affected in state['ir']:
                state['ir'][affected]['stale'] = True
                state['ir'][affected]['stale_reason'] = 'Contract changed: ' + name
        checkpoint()
        commit(new)

    def propose(name, current, issues):
        packet = evidence(name)
        if name in current:
            packet['previous_contract'] = compact_contract(current[name])
        packet['reconciliation_feedback'] = issues
        last_failure = state.get('signature_failures', {}).get(name)
        if last_failure:
            packet['previous_signature_feedback'] = last_failure.get('feedback', '')[:8000]
        adapter.header = ''
        result = attach_abi(batch(client, config, adapter, functions[name], 'signature',
                                 directory / 'functions' / name / 'signature', packet))
        write_json(directory / 'functions' / name / 'last_signature_proposal.json', result)
        # Worklist checkpointing owns the accepted contract update.
        if not result.get('accepted'):
            state.setdefault('signature_failures', {})[name] = result
        else:
            state.setdefault('signature_failures', {}).pop(name, None)
        checkpoint()
        if not result.get('accepted'):
            commit(result)
        return result

    if enhanced and not state.get('initial_reconciliation_done'):
        reconciliation = reconcile_worklist(ordered, contracts, outgoing, propose, max_visits=3, on_change=changed)
        state['reconciliation'].append(reconciliation)
        state['initial_reconciliation_done'] = True
        checkpoint()
        for contract in contracts.values():
            commit(contract)
    if not enhanced:
        state['initial_reconciliation_done'] = True

    # Revisit stale callers when a body-stage contract review changes an earlier hypothesis.
    for sweep in range(3):
        for name in ordered:
            root = directory / 'functions' / name
            previous = state['ir'].get(name)
            if previous and previous.get('accepted') and not previous.get('stale'):
                continue
            if name not in contracts:
                for _ in range(config['max_attempts']):
                    proposal = propose(name, contracts, [])
                    if proposal.get('accepted'):
                        contracts[name] = proposal
                        changed(name, None, proposal)
                    checkpoint()
                    commit(proposal)
                    if proposal.get('accepted') or proposal.get('terminal') or proposal.get('requests', 10) >= config['max_attempts']:
                        break
            if name not in contracts:
                continue
            adapter.header = contracts[name]['contract']['project_header']
            adapter.direct_probes[name], adapter.probe_errors[name] = adapter.parse_probes(contracts[name]['contract'].get('direct_probes', []))
            if previous and previous.get('candidate') and previous.get('stale'):
                checked = adapter.validate(name, functions[name]['source_file'], contracts[name]['signature'],
                                           Path(previous['candidate']), root / f'revalidate_{sweep}', 'ir')
                previous.update(accepted=checked['accepted'], validation=checked, feedback=checked.get('feedback', ''), stale=False)
                checkpoint()
                if previous['accepted']:
                    continue
            for iteration in range(config['max_attempts']):
                packet = evidence(name)
                packet.update(current_contract=compact_contract(contracts[name]), project_header=adapter.header)
                if previous:
                    packet['previous_feedback'] = previous.get('feedback', '')[:10000]
                    candidate = previous.get('candidate')
                    if candidate and Path(candidate).exists():
                        code = Path(candidate).read_text()
                        # Omission is explicit; never truncate a previous function into invalid IR.
                        if len(code) <= 48000:
                            packet['previous_candidate'] = code
                        else:
                            packet['previous_candidate_omitted'] = 'Over 48000 characters; regenerate from complete assembly'
                result = batch(client, config, adapter, functions[name], 'ir', root / 'ir', packet, contracts[name]['signature'])
                if result.get('reason') == 'Shared stage generation budget exhausted':
                    if name not in state['ir']:
                        state['ir'][name] = result
                        checkpoint()
                    break
                state['ir'][name] = result
                checkpoint()
                commit(result)
                previous = result
                if result.get('needs_probe'):
                    probe_packet = evidence(name)
                    probe_packet.update(current_contract=compact_contract(contracts[name]), probe_feedback=result.get('feedback', ''))
                    probes = batch(client, config, adapter, functions[name], 'probes', root / 'probes', probe_packet, contracts[name]['signature'])
                    state.setdefault('probes', {})[name] = probes
                    checkpoint()
                    commit(probes)
                    if probes.get('accepted') and result.get('candidate'):
                        checked = adapter.validate(name, functions[name]['source_file'], contracts[name]['signature'],
                                                   Path(result['candidate']), root / f'probe_check_{sweep}_{iteration}', 'ir')
                        result.update(accepted=checked['accepted'], validation=checked, feedback=checked.get('feedback', ''))
                        checkpoint()
                if result.get('accepted'):
                    break
                if result.get('terminal') or result.get('needs_probe') and not result.get('validation', {}).get('execution_observed') and state.get('probes', {}).get(name, {}).get('requests', 0) >= 3:
                    break
                kind = result.get('validation', {}).get('failure_kind')
                review = enhanced and (kind == 'abi_conflict' or kind == 'runtime' and iteration >= 1)
                if review and state['reviews'].get(name, 0) < 2:
                    state['reviews'][name] = state['reviews'].get(name, 0) + 1
                    old = contracts[name]
                    proposal = propose(name, contracts, [{'kind': 'body_validation_review', 'failure_kind': kind,
                                                          'feedback': result.get('feedback', '')[:6000],
                                                          'note': 'Failure may be in the body; do not change a supported ABI without evidence'}])
                    if proposal.get('accepted') and contract_fingerprint(proposal) != contract_fingerprint(old):
                        contracts[name] = proposal
                        changed(name, old, proposal)
                        reconciliation = reconcile_worklist(ordered, contracts, outgoing, propose, max_visits=1, on_change=changed)
                        state['reconciliation'].append(reconciliation)
                        adapter.header = contracts[name]['contract']['project_header']
                        adapter.direct_probes[name], adapter.probe_errors[name] = adapter.parse_probes(contracts[name]['contract'].get('direct_probes', []))
                    checkpoint()
                    commit(proposal)
                if result.get('requests', 10) >= config['max_attempts']:
                    break
            print(config['arm'], program, name, 'IR', state['ir'][name].get('accepted', False), flush=True)
        if enhanced:
            conflicting = [n for n in ordered if n in contracts and contradictions(
                n, {**contracts[n].get('dependencies', {}), n: contracts[n]['abi']}, contracts, outgoing)]
            if conflicting:
                report = reconcile_worklist(conflicting, contracts, outgoing, propose, max_visits=1, on_change=changed)
                state['reconciliation'].append(report)
                checkpoint()
        if not any(r.get('stale') for r in state['ir'].values()):
            break
    unresolved = {n: contradictions(n, {**c.get('dependencies', {}), n: c['abi']}, contracts, outgoing)
                  for n, c in contracts.items()} if enhanced else {}
    results = []
    for name in ordered:
        recovered = copy.deepcopy(state['ir'].get(name, {'accepted': False, 'reason': 'No accepted signature'}))
        if recovered.get('stale') or unresolved.get(name):
            recovered.update(accepted=False, reason='Unresolved or stale interprocedural contract')
        result = {'name': name, 'signature': contracts.get(name, state.get('signature_failures', {}).get(name, {})),
                  'ir': recovered, 'contract_conflicts': unresolved.get(name, [])}
        results.append(result)
        write_json(directory / 'functions' / name / 'result.json', result)
    write_json(directory / 'results.json', results)
    summary = {'arm': config['arm'], 'program': program, 'targets': len(ordered),
               'accepted_ir': sum(r['ir'].get('accepted', False) for r in results),
               'accepted_c': 0, 'complete_project_recovery': False, 'ir_only': True,
               'unresolved_contracts': sum(bool(v) for v in unresolved.values()), 'abi_proven': False}
    write_json(directory / 'summary.json', summary)
    return summary


def legacy_running(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    try:
        return b'models.binary_decompilation.controlled_ir_experiment' in path.read_bytes()
    except FileNotFoundError:
        return False


def supervise(root, configs, wait_pid=None):
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while wait_pid and legacy_running(wait_pid):
            write_json(root / 'runner_status.json', {'phase': 'waiting_for_legacy_campaign', 'pid': os.getpid(), 'waiting_for_pid': wait_pid})
            time.sleep(30)
        for arm, path in configs.items():
            config = load_config(path)
            workers = {}
            try:
                for program in config['programs']:
                    output = Path(config['output_dir']) / 'retrieval' / program
                    if (output / 'summary.json').exists():
                        continue
                    log = root / 'logs' / (arm + '_' + program + '.log')
                    log.parent.mkdir(exist_ok=True)
                    with log.open('a') as stream:
                        process = subprocess.Popen([sys.executable, '-u', '-m', 'models.binary_decompilation.reconciled_campaign', '--worker-config', str(path),
                                                    '--program', program], stdout=stream, stderr=subprocess.STDOUT)
                    workers[program] = process
                write_json(root / 'runner_status.json', {'phase': arm, 'pid': os.getpid(), 'workers': {n: p.pid for n, p in workers.items()}})
                for name, process in workers.items():
                    if process.wait():
                        raise RuntimeError(arm + '/' + name + ' worker failed')
            finally:
                for process in workers.values():
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                for process in workers.values():
                    process.wait()
        heldout = {arm: evaluate_heldout(path, root / 'heldout' / arm) for arm, path in configs.items()}
        rows = []
        for arm, path in configs.items():
            config = load_config(path)
            for program in config['programs']:
                result = json.loads((Path(config['output_dir']) / 'retrieval' / program / 'results.json').read_text())
                rows.extend({'arm': arm, 'program': program, 'function': r['name'],
                             'accepted_ir': bool(r['ir'].get('accepted'))} for r in result)
        aggregates = {}
        for arm, path in configs.items():
            config = load_config(path)
            arm_rows = [r for r in rows if r['arm'] == arm]
            requests = list(Path(config['output_dir']).glob('retrieval/*/functions/*/*/round_*/attempt_0/request.json'))
            tokens, missing = 0, 0
            for request in requests:
                response_path = request.parent / 'response.json'
                response = json.loads(response_path.read_text()) if response_path.exists() else {}
                usage = response.get('usage') or {}
                if usage.get('total_tokens') is None:
                    missing += 1
                else:
                    tokens += usage['total_tokens']
            accepted = sum(r['accepted_ir'] for r in arm_rows)
            aggregates[arm] = {'accepted_ir': accepted, 'targets': len(arm_rows),
                               'success_rate': accepted / len(arm_rows) if arm_rows else 0,
                               'generation_requests': len(requests),
                               'requested_candidates': sum(json.loads(p.read_text()).get('n', 1) for p in requests),
                               'reported_total_tokens': tokens, 'requests_without_usage': missing}
        paired = {}
        for row in rows:
            paired.setdefault(row['program'] + '::' + row['function'], {})[row['arm']] = row['accepted_ir']
        gains = [name for name, pair in paired.items() if pair.get('reconciled') and not pair.get('baseline')]
        losses = [name for name, pair in paired.items() if pair.get('baseline') and not pair.get('reconciled')]
        write_json(root / 'comparison.json', {'development': rows, 'aggregate': aggregates, 'gains': gains,
                   'losses': losses, 'heldout': heldout,
                   'interpretation': 'One-seed end-to-end workflow comparison, not proof of causality or equivalence'})
        write_json(root / 'runner_status.json', {'phase': 'finished', 'pid': os.getpid(), 'finished_unix': time.time()})


def prepare(root, base_path):
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(base_path)
    project_root = Path(config['source_dir']).resolve().parent.parent
    config.update(ir_only=True, compiler_analysis=False, semantic_analysis=False, guided_decompilation=False,
                  translation_scaffold=False, semantic_checks=False, num_generate=8, max_attempts=10,
                  max_probe_attempts=3, enable_thinking=True, stage_enable_thinking={'signature': False, 'probes': False},
                  seed=1729, input_ceiling=65536, request_timeout=1800, service_health_timeout=60,
                  plan_cache=str(project_root / 'validation/program_analysis_ir_v2/full_A_1729'),
                  retrieval_cache=str(root / 'shared_retrieval'),
                  context_window=260000, max_tokens=260000, max_tokens_policy='remaining_context', stage_max_tokens={})
    config['retrieval'].update(structural_rerank=False, candidate_pool=20, max_examples=2)
    configs = {}
    for arm in ('baseline', 'reconciled'):
        settings = copy.deepcopy(config)
        settings.update(arm=arm, signature_reconciliation=arm == 'reconciled',
                        analysis_mode='compact_interprocedural' if arm == 'reconciled' else 'none',
                        output_dir=str(root / arm))
        path = root / 'configs' / (arm + '.json')
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError('Frozen comparison config changed')
        write_json(path, settings)
        configs[arm] = path
    implementations = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir()
                       if p.suffix in ('.py', '.cpp')}
    identity = root / 'implementation.json'
    if identity.exists() and json.loads(identity.read_text()) != implementations:
        raise ValueError('Implementation changed; use a new comparison root')
    write_json(identity, implementations)
    from models.binary_decompilation.controlled_ir_experiment import HELDOUT
    write_json(root / 'protocol.json', {'arms': ['baseline', 'reconciled'], 'seed': 1729,
               'targets': 20, 'candidates': 8, 'max_rounds_per_target_stage': 10, 'heldout': HELDOUT,
               'baseline': 'Address-order generation from assembly, raw ELF data and RAG; no interprocedural evidence',
               'reconciled': 'Caller/callee contexts, bounded analysis summaries and bidirectional contract revisions',
               'signature_loop': {'initial_visits_per_function': 3, 'body_triggered_reviews': 2,
                                  'shared_signature_round_limit': 10, 'caller_revalidation_sweeps': 3,
                                  'abi_proven': False},
               'comparison_policy': 'Matched validation, RAG, model settings and request caps; one seed, no causal isolation of individual components',
               'legacy_snapshot': str(project_root / 'validation/code_snapshots/pre_signature_reconciliation_20261014')})
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-config', type=Path, default=Path(__file__).with_name('configs') / 'coreutils_binary_candidates8.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--program')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_config:
        run_program(load_config(args.worker_config), args.program)
        return
    if not args.output:
        parser.error('--output is required')
    root = args.output.resolve()
    configs = prepare(root, args.base_config)
    if args.run:
        try:
            supervise(root, configs, args.wait_for_pid)
        except BaseException as error:
            write_json(root / 'runner_status.json', {'phase': 'failed', 'pid': os.getpid(), 'error': str(error)})
            raise
    else:
        print(json.dumps({'prepared': str(root), 'inference_started': False}))


if __name__ == '__main__':
    main()
