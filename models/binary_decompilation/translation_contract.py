"""Binary-derived LLVM scaffolding and conservative unsafe-attribute diagnostics."""

import json
import re
from pathlib import Path

from elftools.elf.elffile import ELFFile


def llvm_bytes(raw):
    return 'c"' + ''.join('\\%02X' % byte for byte in raw) + '"'


def create_contract(binary, facts, directory, public_abi=None):
    literals = []
    with Path(binary).open('rb') as stream:
        elf = ELFFile(stream)
        addresses = sorted({int(a['rip_target'], 16) for a in facts['cfg']['memory_accesses']
                            if a.get('rip_target')})
        for address in addresses:
            section = next((s for s in elf.iter_sections() if s['sh_flags'] & 2 and not s['sh_flags'] & 1
                            and not s['sh_flags'] & 4 and s['sh_type'] != 'SHT_NOBITS'
                            and s['sh_addr'] <= address < s['sh_addr'] + s['sh_size']), None)
            if section is None:
                continue
            offset = address - section['sh_addr']
            raw = section.data()[offset:offset + 4096]
            end = raw.find(b'\0')
            if end < 1:
                continue
            raw = raw[:end + 1]
            # This is a literal hypothesis, not permission to substitute pointer identity.
            if any(b not in (0, 9, 10, 13, 27) and not 32 <= b < 127 for b in raw):
                continue
            symbol = '__decomp_literal_%x' % address
            definition = '@%s = private constant [%d x i8] %s, align 1' % (symbol, len(raw), llvm_bytes(raw))
            literals.append({'address': hex(address), 'symbol': symbol, 'byte_count': len(raw),
                             'definition': definition, 'status': 'heuristic_string_extent',
                             'restriction': 'Use only for value-only string reads; do not replace address identity or mutable globals'})
    result = {'schema_version': 1, 'target_triple': 'x86_64-unknown-linux-gnu',
              'verified_public_declarations': [item['llvm_declaration'] for item in (public_abi or {}).values()
                                               if item.get('llvm_declaration')],
              'optional_literal_definitions': literals,
              'block_labels': {b['start']: 'bb_' + b['start'][2:] for b in facts['cfg']['blocks']},
              'requirements': ['Generate the function body yourself; no lifted instruction bodies are supplied',
                               'Use exact ELF dependency names and verified public declarations',
                               'Use named SSA values or finite reusable stack slots',
                               'Do not add nsw/nuw/exact/inbounds or ABI/alias attributes without independently verified assumptions',
                               'Preserve observable stores, branch predicates, integer widths and call effects',
                               'Use ; asm-block: 0xADDRESS comments to map emitted blocks to the assembly',
                               'Optional literal definitions must be copied exactly if used; no substitution of pointer identity']}
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / 'translation_contract.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def ir_tokens(text):
    # Strip LLVM quoted strings, comments, and identifiers before matching attributes.
    return re.sub(r'"(?:\\.|[^"\\])*"|;[^\n]*|[%@][-a-zA-Z$._0-9]+', ' ', text)


def inspect_ir(text, contract=None):
    declarations = set((contract or {}).get('verified_public_declarations', []))
    unverified = '\n'.join(line for line in text.splitlines() if line.strip() not in declarations)
    unsupported = sorted(set(re.findall(r'\b(?:nsw|nuw|exact|inbounds|noalias|nonnull|noundef|dereferenceable)\b', ir_tokens(unverified))))
    errors = ['Unproved LLVM semantic strengthening: ' + ', '.join(unsupported)] if unsupported else []
    if re.search(r'@(?:"llvm\.assume"|llvm\.assume)\s*\(', unverified):
        errors.append('llvm.assume requires independently proved preconditions')
    for line in ir_tokens(unverified).splitlines():
        if re.search(r'\b(?:load|store)\b', line):
            alignment = re.search(r'\balign\s+(\d+)', line)
            if alignment and int(alignment.group(1)) > 1:
                errors.append('Use align 1 for memory operations without a proved alignment contract')
    if contract:
        lines = {line.strip() for line in text.splitlines()}
        for literal in contract['optional_literal_definitions']:
            if re.search(r'@' + re.escape(literal['symbol']) + r'\b', text) and literal['definition'] not in lines:
                errors.append('Use the byte-exact definition: ' + literal['definition'])
    mapping = re.findall(r';\s*asm-block:\s*(0x[0-9a-fA-F]+)', text)
    return {'schema_version': 1, 'accepted': not errors, 'errors': errors,
            'block_mapping': mapping, 'mapping_status': 'model_asserted_not_verified',
            'semantic_equivalence': 'unknown'}
