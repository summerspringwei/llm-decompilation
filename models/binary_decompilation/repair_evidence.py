"""Localize compiler diagnostics through explicitly unverified model block maps."""

import re

from models.binary_decompilation.semantic_analysis import backward_slice


def compiler_slice(facts, llvm_ir, diagnostic):
    match = re.search(r'[^\s:]+\.ll:(\d+):\d+:', diagnostic)
    if not match:
        return {'localized': False, 'reason': 'No LLVM source location; runtime output differences alone do not identify a block'}
    lines = llvm_ir.splitlines()
    line_number = int(match.group(1))
    if not 1 <= line_number <= len(lines):
        return {'localized': False, 'reason': 'Diagnostic line is outside candidate'}
    markers = re.findall(r';\s*asm-block:\s*(0x[0-9a-fA-F]+)', '\n'.join(lines[:line_number]))
    if not markers:
        return {'localized': False, 'reason': 'No candidate block mapping at the diagnostic'}
    address = int(markers[-1], 16)
    block = next((b for b in facts['cfg']['blocks'] if int(b['start'], 16) == address), None)
    if not block:
        return {'localized': False, 'reason': 'Candidate mapped to an unknown assembly block'}
    roots = [r['id'] for r in facts['pcode'] if address <= int(r['address'], 16) <= int(block['end'], 16)]
    sliced = backward_slice(roots, facts['dataflow']['dependencies'], limit=120)
    return {'localized': True, 'status': 'advisory_model_mapping_not_proof', 'llvm_line': line_number,
            'assembly_block': block['start'], 'slice': [r for r in facts['pcode'] if r['id'] in sliced['definitions']],
            'truncated': sliced['truncated'],
            'instruction': 'Repair the diagnosed IR region; revalidate the complete function and preserve other observable effects'}
