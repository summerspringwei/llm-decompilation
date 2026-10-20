"""Caller/callee contract reconciliation, without accepting source types as truth."""

import copy
import hashlib
import json
from pathlib import Path
import re
import shlex

import capstone as cs
from capstone import x86_const as x86

from models.binary_decompilation.binary_project import BinaryCoreutilsAdapter
from models.binary_decompilation.native_validation import elf_symbols
from models.binary_decompilation.project_pipeline import run, write_json
from models.binary_decompilation.semantic_analysis import function_bytes


def abi_identity(abi):
    return {k: v for k, v in abi.items() if k not in ('defined', 'calls')}


class ABIReader:
    """Use the matching LLVM parser, not regex parsing of textual IR types."""

    def __init__(self, clang, directory):
        self.clang, self.directory = clang, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name('inspect_llvm_abi.cpp')
        version = run([clang, '--version'])['stdout']
        key = hashlib.sha256(source.read_bytes() + version.encode()).hexdigest()[:16]
        self.executable = self.directory / ('inspect_abi_' + key)
        if not self.executable.exists():
            config = str(Path(clang).with_name('llvm-config'))
            flags = run([config, '--cxxflags', '--ldflags', '--libs', 'core', 'irreader', 'support', '--system-libs'])
            if flags['returncode']:
                raise RuntimeError(flags['stderr'])
            result = run([str(Path(clang).with_name('clang++')), str(source),
                          *shlex.split(flags['stdout']), '-o', str(self.executable)], timeout=180)
            write_json(self.directory / 'abi_reader_build.json', result)
            if result['returncode']:
                raise RuntimeError(result['stderr'])

    def read(self, path):
        result = run([str(self.executable), str(path)])
        if result['returncode']:
            raise ValueError(result['stderr'])
        return json.loads(result['stdout'])

    def contract(self, name, header, declaration, project_names, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        source, ir = directory / 'contract.c', directory / 'contract.ll'
        source.write_text(header + '\n' + declaration + '\n')
        ast = run([self.clang, '-D_GNU_SOURCE', '-std=gnu17', '-fsyntax-only', '-Xclang', '-ast-dump=json', str(source)])
        if ast['returncode']:
            raise ValueError(ast['stderr'])
        nodes = json.loads(ast['stdout']).get('inner', [])
        declared = {n['name'] for n in nodes if n.get('kind') == 'FunctionDecl'}
        names = sorted((set(project_names) | {name}) & declared)
        if name not in names or any(not re.fullmatch(r'[A-Za-z_]\w*', n) for n in names):
            raise ValueError('Contract must declare the exact C target identifier')
        source.write_text(source.read_text() + '\nvoid *__reconciliation_anchors[] = {' +
                          ','.join('(void *)&' + n for n in names) + '};\n')
        result = run([self.clang, '-D_GNU_SOURCE', '-std=gnu17', '-O0', '-S', '-emit-llvm', str(source), '-o', str(ir)])
        write_json(directory / 'emit_contract.json', result)
        if result['returncode']:
            raise ValueError(result['stderr'])
        functions = self.read(ir)
        if name not in functions:
            raise ValueError('Target assembler alias differs from the binary target')
        return {n: functions[n] for n in names if n in functions}


def extract_call_sites(binary, functions):
    """Direct call-site slices include argument setup and return use, not whole callers."""
    symbols = elf_symbols(binary)
    names_by_address = {v['address']: name for name, v in symbols.items()
                        if isinstance(v, dict) and v.get('kind') == 'STT_FUNC'}
    target_names = {f['name'] for f in functions}
    names_by_address.update({symbols[n]['address']: n for n in target_names})
    sites, outgoing = {n: [] for n in target_names}, {n: [] for n in target_names}
    decoder = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    decoder.detail = True
    for function in functions:
        code, entry = function_bytes(binary, function['name'])
        instructions = list(decoder.disasm(code, entry))
        for index, ins in enumerate(instructions):
            if not (ins.group(cs.CS_GRP_CALL) or ins.mnemonic == 'jmp') or not ins.operands:
                continue
            if ins.operands[0].type != x86.X86_OP_IMM:
                continue
            callee = names_by_address.get(ins.operands[0].imm)
            if callee not in target_names or callee == function['name'] and ins.mnemonic == 'jmp':
                continue
            record = {'caller': function['name'], 'callee': callee, 'address': hex(ins.address),
                      'tail_call': ins.mnemonic == 'jmp',
                      'before': [f'{i.address:x}: {i.mnemonic} {i.op_str}' for i in instructions[max(0, index-16):index]],
                      'after': [f'{i.address:x}: {i.mnemonic} {i.op_str}' for i in instructions[index+1:index+9]],
                      'status': 'local_instruction_window_not_a_proven_path'}
            sites[callee].append(record)
            outgoing[function['name']].append(callee)
    return {'incoming': sites, 'outgoing': {n: sorted(set(v)) for n, v in outgoing.items()},
            'limitations': ['Indirect callers are not resolved by this extractor',
                            'Call-site windows may cross branches and are observations, not hard argument-count constraints']}


def contradictions(name, proposed, contracts, outgoing):
    """Detect incompatible declarations across actual caller/callee edges."""
    if name not in proposed:
        return [{'kind': 'missing_target', 'function': name}]
    issues = []
    for callee in outgoing.get(name, []):
        target = contracts.get(callee, {}).get('abi')
        declared = proposed.get(callee)
        if target and declared and abi_identity(target) != abi_identity(declared):
            issues.append({'kind': 'callee_declaration_conflict', 'caller': name, 'callee': callee,
                           'caller_declaration': abi_identity(declared), 'callee_contract': abi_identity(target)})
    for caller, contract in contracts.items():
        if name not in outgoing.get(caller, []):
            continue
        declared = contract.get('dependencies', {}).get(name)
        if declared and abi_identity(declared) != abi_identity(proposed[name]):
            issues.append({'kind': 'caller_declaration_conflict', 'caller': caller, 'callee': name,
                           'caller_declaration': abi_identity(declared), 'callee_contract': abi_identity(proposed[name])})
    return issues


def compact_contract(contract):
    return {'signature': contract.get('signature'), 'abi': contract.get('abi'),
            'dependencies': contract.get('dependencies', {}),
            'abi_verified': False, 'status': contract.get('status', 'hypothesis')}


def contract_fingerprint(result):
    relevant = {'abi': result.get('abi'), 'dependencies': result.get('dependencies'),
                'header': result.get('contract', {}).get('project_header', '')}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def reconcile_worklist(names, contracts, outgoing, propose, max_visits=3, on_change=None):
    """Revisit both ends of changed contracts; cycles/budget exhaustion remain unresolved."""
    queue = list(names)
    visits = {n: 0 for n in names}
    seen = {n: {contract_fingerprint(contracts[n])} if n in contracts else set() for n in names}
    events = []
    while queue:
        name = queue.pop(0)
        if visits[name] >= max_visits:
            continue
        visits[name] += 1
        previous = contracts.get(name)
        issues = contradictions(name, {name: previous['abi'], **previous.get('dependencies', {})}, contracts, outgoing) if previous and previous.get('abi') else []
        result = propose(name, contracts, issues)
        if not result or not result.get('accepted'):
            events.append({'function': name, 'event': 'proposal_failed', 'visit': visits[name]})
            if previous is None and visits[name] < max_visits:
                queue.append(name)
            continue
        fingerprint = contract_fingerprint(result)
        if previous and fingerprint == contract_fingerprint(previous):
            events.append({'function': name, 'event': 'unchanged', 'visit': visits[name]})
            continue
        if fingerprint in seen[name]:
            events.append({'function': name, 'event': 'cycle_detected', 'visit': visits[name]})
            continue
        seen[name].add(fingerprint)
        contracts[name] = result
        events.append({'function': name, 'event': 'contract_changed', 'visit': visits[name], 'fingerprint': fingerprint})
        if on_change:
            on_change(name, previous, result)
        neighbors = set(outgoing.get(name, [])) | {n for n, deps in outgoing.items() if name in deps}
        for neighbor in names:
            if neighbor in neighbors and neighbor not in queue and visits[neighbor] < max_visits:
                queue.append(neighbor)
    unresolved = {name: contradictions(name, {**c.get('dependencies', {}), name: c['abi']}, contracts, outgoing)
                  for name, c in contracts.items() if c.get('abi')}
    unresolved = {n: issues for n, issues in unresolved.items() if issues}
    missing = sorted(set(names) - set(contracts))
    return {'events': events, 'visits': visits, 'unresolved': unresolved, 'missing_contracts': missing,
            'declaration_consistent': not unresolved and not missing,
            'abi_proven': False, 'note': 'Declaration agreement is not proof of binary ABI correctness'}


class ReconciliationAdapter(BinaryCoreutilsAdapter):
    def __init__(self, config, program, output, reader, functions):
        super().__init__(config, program, output)
        self.reader = reader
        self.project_names = set(functions)
        self.contracts = {}
        self.outgoing = {}
        self.enforce_dependencies = config.get('signature_reconciliation', False)

    def validate(self, name, source_file, signature, candidate, directory, stage):
        if stage == 'signature':
            result = super().validate(name, source_file, signature, candidate, directory, stage)
            if not result['accepted']:
                return result
            functions = self.reader.contract(name, self.header, signature, self.project_names, directory / 'abi')
            result['abi'] = functions[name]
            result['dependencies'] = {n: abi for n, abi in functions.items() if n != name}
            if self.enforce_dependencies:
                issues = contradictions(name, functions, self.contracts, self.outgoing)
                result['contract_conflicts'] = issues
                result['probe_errors'] = [*result.get('probe_errors', []), *issues]
            return result
        if stage == 'ir':
            try:
                functions = self.reader.read(candidate)
            except ValueError as error:
                return {'accepted': False, 'failure_kind': 'llvm_parse', 'feedback': str(error)}
            own = self.contracts.get(name, {}).get('abi')
            conflicts = []
            if own and (name not in functions or abi_identity(functions[name]) != abi_identity(own)):
                conflicts.append({'target': name, 'expected': own, 'actual': functions.get(name)})
            if self.enforce_dependencies:
                for callee, contract in self.contracts.items():
                    if callee != name and callee in functions and contract.get('abi') and abi_identity(functions[callee]) != abi_identity(contract['abi']):
                        conflicts.append({'callee': callee, 'expected': contract['abi'], 'actual': functions[callee]})
                for call in functions.get(name, {}).get('calls', []):
                    expected = self.contracts.get(call['callee'], {}).get('abi')
                    if expected and abi_identity(call['abi']) != abi_identity(expected):
                        conflicts.append({'call_site': call['callee'], 'expected': expected, 'actual': call['abi']})
            if conflicts:
                write_json(directory / 'abi_conflicts.json', conflicts)
                return {'accepted': False, 'failure_kind': 'abi_conflict', 'repair_contract': True,
                        'feedback': json.dumps(conflicts), 'abi_verified': False}
        return super().validate(name, source_file, signature, candidate, directory, stage)
