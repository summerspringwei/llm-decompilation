"""Bounded scalar leaf equivalence with VEX and concrete counterexample replay.

This intentionally does not model arbitrary library calls, pointers, or globals.
Unknown is not a successful equivalence result and never upgrades test acceptance.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from models.binary_decompilation.semantic_analysis import function_bytes


def supported_leaf(code, entry):
    import capstone
    from capstone import x86_const as x86
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoder.detail = True
    instructions = list(decoder.disasm(code, entry))
    addresses = {i.address for i in instructions}
    allowed = {"nop", "endbr64", "mov", "movabs", "movzx", "movsx", "movsxd", "lea", "ret",
               "add", "sub", "imul", "and", "or", "xor", "not", "neg", "inc", "dec", "cmp", "test",
               "shl", "sal", "shr", "sar", "rol", "ror", "cdqe", "cwde", "cdq", "cqo"}
    if sum(i.size for i in instructions) != len(code):
        return "Incomplete decoding"
    for ins in instructions:
        if ins.group(capstone.CS_GRP_CALL):
            return "Calls require external-effect models"
        if ins.mnemonic not in allowed and not (ins.mnemonic.startswith('j') or ins.mnemonic.startswith('set') or ins.mnemonic.startswith('cmov')):
            return "Unsupported instruction: " + ins.mnemonic
        if ins.mnemonic not in ('lea', 'nop') and any(o.type == x86.X86_OP_MEM for o in ins.operands):
            return "Memory operands require object/state correspondence"
        if ins.group(capstone.CS_GRP_JUMP) and (ins.operands[0].type != x86.X86_OP_IMM or ins.operands[0].imm not in addresses):
            return "Unresolved transfer or tail call"
        if ins.mnemonic == 'ret' and ins.operands:
            return "Non-SysV return"
        if any(ins.reg_name(r).startswith(('xmm', 'ymm', 'zmm', 'st')) for r in ins.regs_access()[0] + ins.regs_access()[1]):
            return "Non-integer register domain"
        if ins.mnemonic != 'ret' and x86.X86_REG_RSP in ins.regs_access()[1]:
            return "Stack-frame semantics outside scalar leaf domain"
    return None


def compare_scalar(original, candidate, name, candidate_name=None, return_bits=64, timeout=30, path_limit=64):
    import angr
    import claripy

    started = time.monotonic()
    if return_bits not in (8, 16, 32, 64):
        return {'status': 'unknown', 'reason': 'Unsupported return ABI'}
    inputs = {r: claripy.BVS('entry_' + r, 64, explicit_name=True) for r in
              ('rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15')}
    sentinel, stack = 0x7fff0000, 0x7ffe0000

    def execute(path, symbol, concrete=None):
        code, entry = function_bytes(path, symbol)
        unsupported = supported_leaf(code, entry)
        if unsupported:
            raise ValueError(unsupported)
        project = angr.load_shellcode(code, 'amd64', load_address=entry)
        state = project.factory.blank_state(addr=entry)
        state.solver._solver.timeout = int(timeout * 1000)
        for reg, value in inputs.items():
            setattr(state.regs, reg, concrete[reg] if concrete is not None else value)
        state.regs.rsp = stack
        state.memory.store(stack, sentinel, size=8, endness=project.arch.memory_endness)
        manager = project.factory.simulation_manager(state)
        done = []
        for _ in range(256):
            if time.monotonic() - started > timeout:
                raise ValueError('Symbolic time budget exhausted')
            manager.move(from_stash='active', to_stash='returned', filter_func=lambda s: s.addr == sentinel)
            done.extend(manager.stashes.pop('returned', []))
            if not manager.active:
                break
            if len(manager.active) + len(done) > path_limit:
                raise ValueError('Symbolic path budget exhausted')
            if any(s.solver.symbolic(s.regs.rip) or not entry <= s.addr < entry + len(code) for s in manager.active):
                raise ValueError('Execution escaped supported function')
            manager.step()
            if manager.errored or manager.unconstrained or manager.deadended:
                raise ValueError('Incomplete symbolic execution')
        if manager.active or not done:
            raise ValueError('Incomplete bounded execution')
        allowed_variables = set().union(*(v.variables for v in inputs.values()))
        for state in done:
            expressions = [state.regs.rax, *state.solver.constraints]
            if any(e.variables - allowed_variables for e in expressions):
                raise ValueError('Unmodeled initial flag or register dependency')
        return done

    try:
        left, right = execute(original, name), execute(candidate, candidate_name or name)
        solver = claripy.Solver(timeout=int(timeout * 1000))
        left_cover = claripy.Or(*(claripy.And(*s.solver.constraints) for s in left))
        right_cover = claripy.Or(*(claripy.And(*s.solver.constraints) for s in right))
        if solver.satisfiable(extra_constraints=[claripy.Not(claripy.And(left_cover, right_cover))]):
            return {'status': 'unknown', 'reason': 'Not all scalar input states terminate within the domain'}
        for a in left:
            for b in right:
                mismatch = [*a.solver.constraints, *b.solver.constraints,
                            a.regs.rax[return_bits - 1:0] != b.regs.rax[return_bits - 1:0]]
                query = claripy.Solver(timeout=int(timeout * 1000))
                query.add(mismatch)
                if not query.satisfiable():
                    continue
                model = query.batch_eval(list(inputs.values()), 1)[0]
                values = dict(zip(inputs, model))
                actual_left = execute(original, name, values)
                actual_right = execute(candidate, candidate_name or name, values)
                expected = actual_left[0].solver.eval(actual_left[0].regs.rax[return_bits - 1:0])
                actual = actual_right[0].solver.eval(actual_right[0].regs.rax[return_bits - 1:0])
                if len(actual_left) != 1 or len(actual_right) != 1 or expected == actual:
                    return {'status': 'unknown', 'reason': 'Counterexample replay was inconclusive'}
                return {'status': 'counterexample', 'inputs': values, 'expected_return': expected,
                        'actual_return': actual, 'return_bits': return_bits,
                        'replay': 'concrete VEX execution of both machine-code bodies',
                        'reachability': 'unrestricted scalar function entry; caller reachability not established'}
        return {'status': 'bounded_equivalent', 'return_bits': return_bits,
                'paths': [len(left), len(right)],
                'assumptions': ['Scalar call-free domain with no explicit memory access',
                                'Return ABI width supplied by candidate, not independently proved',
                                'Comparison covers return bits only; no claim about entire application']}
    except Exception as error:
        return {'status': 'unknown', 'reason': type(error).__name__ + ': ' + str(error)}


def check_candidate(binary, library, name, return_bits, directory, timeout=30):
    path = Path(directory) / 'semantic_validation.json'
    command = [sys.executable, '-m', __name__, '--original', str(binary), '--candidate', str(library),
               '--name', name, '--candidate-name', '__recovered_' + name,
               '--return-bits', str(return_bits), '--timeout', str(timeout), '--output', str(path)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 15)
        if process.returncode == 0 and path.exists():
            return json.loads(path.read_text())
        result = {'status': 'unknown', 'reason': 'Semantic worker failed', 'stderr': process.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        result = {'status': 'unknown', 'reason': 'Semantic worker wall-time limit'}
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--candidate-name')
    parser.add_argument('--return-bits', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_scalar(args.original, args.candidate, args.name, args.candidate_name,
                            args.return_bits, args.timeout)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
