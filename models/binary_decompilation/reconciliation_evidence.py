"""Bounded evidence packets and a strict no-analysis generation boundary."""

import copy
import json
import re

from models.binary_decompilation.prompt_budget import fit_prompt
from models.binary_decompilation.signature_reconciliation import compact_contract


SIGNATURE = '''Infer the optimized SysV AMD64 function ABI from the supplied binary evidence.
Return ONLY JSON with c_declaration, project_header, argument_evidence, return_evidence,
memory_layouts, uncertainties, and direct_probes. The C declaration must name the exact
target and end in a semicolon. project_header may include standard system headers,
inferred type definitions, and external declarations only, never function bodies.
Treat register values as observations, not proof of argument count. Distinguish pointer
use from integer values and respect partial-register widths. Preserve exact ELF names,
using __asm__("symbol") for C identifiers when necessary. Do not copy retrieved bodies.
If previous_contract and reconciliation_feedback are present, reconsider its ABI and
dependency declarations, not just formatting. Address conflicts using machine evidence;
mutual declaration agreement alone does not establish correctness. State uncertainties.
Optional direct_probes must be an array of at most eight safe isolated calls. A probe
has return_type and argument_types using Frida native types (void,int,uint,uint64,pointer),
arguments, return_kind (integer or cstring), optional setup, and memory_outputs.
Arguments may be numbers, {string:text}, {buffer:byte_count,id:name,bytes:byte_array},
{ref:name}, or {null:true}. Buffers are at most 4096 bytes; references name earlier buffers.
Setup may call only stat/lstat/memcpy/memset using symbol,return_type,argument_types,arguments.
memory_outputs entries have id,size. Do not fabricate arbitrary pointers or unsafe objects.
Use an empty direct_probes array when a safe setup cannot be established.
'''

IR = '''Translate the target assembly into a complete LLVM 18 module defining that function.
Return only a fenced llvm code block. Use opaque pointers and the supplied current ABI.
Declare original application dependencies and globals, do not define or replace their storage.
Preserve exact ELF names, integer widths, sign extension, branch predicates, memory effects,
and library call conventions. Use named SSA values and labels, or finite reusable stack slots.
String array lengths must count decoded bytes, including terminators; hex escapes have two digits.
Use GEP instructions for runtime addresses, not constant expressions containing SSA values.
Do not invent nsw/nuw/inbounds assumptions. Do not delegate to the original target.
Repair according to compiler/runtime feedback. A proposed ABI is a hypothesis: if an ABI conflict
is reported, preserve the evidence and make dependency declarations consistent with the current
contract. The outer reconciliation loop can revise that contract when evidence warrants it.
'''

PROBES = SIGNATURE[SIGNATURE.index('Optional direct_probes'):].replace('Optional direct_probes', 'Return ONLY JSON with direct_probes')


def raw_evidence(function, plan, examples):
    assembly = function.get('symbolized_disassembly') or function['assembly']
    referenced = set(re.findall(r'<([^>+]+)(?:\+[^>]+)?>', assembly))
    addresses = {int(s, 16) for s in re.findall(r'#\s*([0-9a-fA-F]+)', assembly)}
    objects = [o for o in plan['binary_data'].get('objects', []) if o['name'] in referenced]
    addresses.update(r['addend'] for o in objects for r in o.get('relocations', []) if isinstance(r.get('addend'), int))
    strings = []
    for s in plan['binary_data'].get('referenced_strings', []):
        address = s.get('address', 0)
        address = int(address, 0) if isinstance(address, str) else address
        if address in addresses:
            strings.append(s)
    examples = [{k: v for k, v in example.items() if k in ('score', 'collection', 'index', 'fname', 'c', 'llvm_ir')}
                for example in examples]
    return {'target': function['name'], 'assembly': assembly,
            'raw_elf_data': {'objects': objects, 'strings': strings}, 'retrieved_examples': examples,
            'policy': 'No application source, reference IR, or decompiler-generated C is supplied'}


def compact_analysis(facts, name, sites, contracts, runtime, outgoing):
    flow = facts['dataflow']
    neighbors = set(outgoing.get(name, [])) | {caller for caller, callees in outgoing.items() if name in callees}
    memory = facts['memory'].get('layout_constraints', {})
    return {
        'analysis_summary': {'status': flow['status'], 'assumptions': flow['assumptions'],
                             'entry_registers_read': flow['entry_argument_registers_read'],
                             'memory_layout_constraints': {k: v[:24] for k, v in list(memory.items())[:12]},
                             'block_count': len(facts['cfg']['blocks']),
                             'unresolved_jumps': facts['cfg'].get('unresolved_indirect_jumps', []),
                             'limitations': flow['limitations']},
        'caller_sites': sites['incoming'].get(name, [])[:12],
        'caller_sites_omitted': max(0, len(sites['incoming'].get(name, [])) - 12),
        'neighbor_contracts': {n: compact_contract(contracts[n]) for n in sorted(neighbors) if n in contracts},
        'entry_register_samples': runtime.get('entry_register_samples', {}).get(name, [])[:2],
    }


def pack(config, instruction, evidence):
    """Cap auxiliary evidence while retaining complete target assembly and reporting omissions."""
    packet = copy.deepcopy(evidence)
    dropped = []
    # A separate prompt ceiling preserves output room and prevents auxiliary-evidence inflation.
    budget_config = {**config, 'context_window': min(config['context_window'], config.get('input_ceiling', 65536) + 65),
                     'max_tokens_policy': 'remaining_context'}
    while True:
        try:
            _, info = fit_prompt(budget_config, instruction, packet)
            # fit_prompt can drop retrieval internally; apply the same omissions to the saved packet.
            for example in info['removed_retrieval_examples']:
                if packet.get('retrieved_examples'):
                    packet['retrieved_examples'].pop()
                    dropped.append({'retrieved_example': example})
            return packet, {'prompt_tokens': info['prompt_tokens'], 'omitted': dropped,
                            'complete_target_assembly': True, 'input_ceiling': config.get('input_ceiling', 65536)}
        except ValueError:
            removed = False
            for key in ('retrieved_examples', 'caller_sites', 'entry_register_samples'):
                if packet.get(key):
                    packet[key].pop()
                    dropped.append(key)
                    removed = True
                    break
            if removed:
                continue
            if packet.get('analysis_summary'):
                packet.pop('analysis_summary')
                dropped.append('analysis_summary')
                continue
            if packet.get('neighbor_contracts'):
                key = sorted(packet['neighbor_contracts'])[-1]
                packet['neighbor_contracts'].pop(key)
                dropped.append('neighbor_contract:' + key)
                continue
            if packet.get('previous_candidate'):
                packet.pop('previous_candidate')
                packet['previous_candidate_omitted'] = 'Prompt budget; complete target assembly retained'
                dropped.append('previous_candidate')
                continue
            raise ValueError('Essential target/contract evidence exceeds input ceiling; no assembly was truncated')


def assert_baseline_boundary(evidence):
    forbidden = {'analysis_summary', 'caller_sites', 'neighbor_contracts', 'entry_register_samples',
                 'compiler_cfg', 'semantic_evidence', 'callee_contracts', 'ghidra_cfg', 'runtime_samples'}
    leaked = forbidden & set(evidence)
    if leaked:
        raise ValueError('No-analysis baseline leaked evidence: ' + ', '.join(sorted(leaked)))
