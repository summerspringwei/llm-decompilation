"""Compiler-style facts from ELF bytes and LLVM passes, never source bodies."""

import hashlib
import json
from pathlib import Path
import re
import struct

import capstone as cs
from capstone import x86_const as x86
from elftools.elf.elffile import ELFFile
import networkx as nx

from models.binary_decompilation.native_validation import elf_symbols
from models.binary_decompilation.project_pipeline import run, write_json


def cfg_facts(blocks, entry):
    graph = nx.DiGraph()
    graph.add_nodes_from(b["start"] for b in blocks)
    for block in blocks:
        graph.add_edges_from((block["start"], dest) for dest in block["successors"] if dest in graph)
    if entry not in graph:
        return {"immediate_dominators": {}, "natural_loops": [], "unreachable_blocks": sorted(graph)}
    dom = nx.immediate_dominators(graph, entry)

    def dominates(header, node):
        while node in dom:
            if node == header:
                return True
            if dom[node] == node:
                break
            node = dom[node]
        return False

    loops = []
    for tail, head in graph.edges:
        if tail in dom and dominates(head, tail):
            members, pending = {head, tail}, [tail] if head != tail else []
            while pending:
                for pred in graph.predecessors(pending.pop()):
                    if pred in dom and pred not in members:
                        members.add(pred)
                        pending.append(pred)
            loops.append({"header": head, "latch": tail, "blocks": sorted(members)})
    return {"immediate_dominators": dom, "natural_loops": loops,
            "unreachable_blocks": sorted(set(graph) - set(dom)),
            "cyclic_components": [sorted(s) for s in nx.strongly_connected_components(graph)
                                  if len(s) > 1 or any(graph.has_edge(n, n) for n in s)]}


def analyze_bytes(code, entry, symbols=None, resolved_jumps=None):
    symbols = symbols or {}
    resolved_jumps = resolved_jumps or {}
    decoder = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    decoder.detail = True
    insns = list(decoder.disasm(code, entry))
    addresses = {i.address for i in insns}
    leaders = {entry}
    unresolved, calls, accesses = [], [], []
    for ins in insns:
        jump = ins.group(cs.CS_GRP_JUMP)
        terminal = jump or ins.group(cs.CS_GRP_RET) or ins.mnemonic in ("ud2", "hlt")
        immediate = ins.operands[0].imm if ins.operands and ins.operands[0].type == x86.X86_OP_IMM else None
        if jump:
            if immediate in addresses:
                leaders.add(immediate)
            elif ins.address in resolved_jumps:
                leaders.update(resolved_jumps[ins.address])
            elif immediate is None:
                unresolved.append(hex(ins.address))
        if terminal and ins.address + ins.size in addresses:
            leaders.add(ins.address + ins.size)
        if ins.group(cs.CS_GRP_CALL) or jump and immediate is not None and immediate not in addresses:
            calls.append({"address": hex(ins.address), "target": hex(immediate) if immediate is not None else None,
                          "name": symbols.get(immediate), "tail_call_candidate": bool(jump),
                          "instruction": ins.mnemonic + " " + ins.op_str})
        for op in ins.operands:
            if op.type == x86.X86_OP_MEM:
                m = op.mem
                accesses.append({"instruction_address": hex(ins.address), "base": ins.reg_name(m.base),
                                 "index": ins.reg_name(m.index), "scale": m.scale, "offset": m.disp,
                                 "width_bytes": op.size, "read": bool(op.access & cs.CS_AC_READ),
                                 "write": bool(op.access & cs.CS_AC_WRITE),
                                 "address_only": ins.mnemonic == "lea",
                                 "rip_target": hex(ins.address + ins.size + m.disp) if m.base == x86.X86_REG_RIP else None})
    groups = []
    for ins in insns:
        if not groups or ins.address in leaders:
            groups.append([])
        groups[-1].append(ins)
    blocks = []
    for group in groups:
        last = group[-1]
        successors = []
        jump = last.group(cs.CS_GRP_JUMP)
        if jump and last.operands[0].type == x86.X86_OP_IMM and last.operands[0].imm in addresses:
            successors.append(hex(last.operands[0].imm))
        elif last.address in resolved_jumps:
            successors.extend(hex(a) for a in resolved_jumps[last.address])
        if not last.group(cs.CS_GRP_RET) and last.mnemonic not in ("jmp", "ljmp", "ud2", "hlt"):
            if last.address + last.size in addresses:
                successors.append(hex(last.address + last.size))
        blocks.append({"start": hex(group[0].address), "end": hex(last.address),
                       "successors": sorted(set(successors)), "terminator": last.mnemonic + " " + last.op_str})
    return {"entry": hex(entry), "decoded_bytes": sum(i.size for i in insns), "symbol_bytes": len(code),
            "complete_linear_decode": sum(i.size for i in insns) == len(code),
            "unresolved_indirect_jumps": unresolved, "blocks": blocks,
            **cfg_facts(blocks, hex(entry)), "calls": calls, "memory_accesses": accesses,
            "limitations": ["Linear symbol decode may include padding or embedded data",
                            "No indirect jump-table resolution; dominators/loops cover known edges only",
                            "Memory bases are instruction-local registers, not recovered source variables"]}


def relative_jump_tables(code, entry, read_memory):
    """Recognize bounded local RIP-base + signed-i32 table dispatches only."""
    decoder = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    decoder.detail = True
    insns = list(decoder.disasm(code, entry))
    addresses = {i.address for i in insns}
    tables = []

    def family(reg):
        name = decoder.reg_name(reg)
        aliases = {'rax': ('eax', 'ax', 'al', 'ah'), 'rbx': ('ebx', 'bx', 'bl', 'bh'),
                   'rcx': ('ecx', 'cx', 'cl', 'ch'), 'rdx': ('edx', 'dx', 'dl', 'dh'),
                   'rsi': ('esi', 'si', 'sil'), 'rdi': ('edi', 'di', 'dil'),
                   'rsp': ('esp', 'sp', 'spl'), 'rbp': ('ebp', 'bp', 'bpl')}
        return next((full for full, parts in aliases.items() if name in parts),
                    re.sub(r'^(r[0-9]+)[dwb]$', r'\1', name))

    for pos, jump in enumerate(insns):
        if jump.mnemonic != 'jmp' or jump.operands[0].type != x86.X86_OP_REG:
            continue
        target = jump.operands[0].reg
        for load_pos in range(max(0, pos - 8), pos):
            load = insns[load_pos]
            if load.mnemonic != 'movsxd' or load.operands[0].reg != target or load.operands[1].type != x86.X86_OP_MEM:
                continue
            memory = load.operands[1].mem
            if not memory.base or not memory.index or memory.scale != 4 or memory.disp or memory.segment:
                continue
            between = insns[load_pos + 1:pos]
            adds = [i for i in between if i.mnemonic == 'add' and len(i.operands) == 2
                    and all(o.type == x86.X86_OP_REG for o in i.operands)
                    and i.operands[0].reg == target and i.operands[1].reg == memory.base]
            if len(adds) != 1 or any(i.group(cs.CS_GRP_CALL) or i.group(cs.CS_GRP_JUMP) or
                                     (any(family(r) == family(target) for r in i.regs_access()[1]) and i is not adds[0]) for i in between):
                continue
            bases = [i for i in insns[max(0, load_pos - 16):load_pos]
                     if i.mnemonic == 'lea' and i.operands[0].reg == memory.base
                     and i.operands[1].type == x86.X86_OP_MEM and i.operands[1].mem.base == x86.X86_REG_RIP]
            if not bases:
                continue
            base_ins = bases[-1]
            table = base_ins.address + base_ins.size + base_ins.operands[1].mem.disp
            if any(i.group(cs.CS_GRP_CALL) or any(family(r) == family(memory.base) for r in i.regs_access()[1])
                   for i in insns if base_ins.address < i.address < jump.address):
                continue
            guards = []
            for k in range(max(0, load_pos - 16), load_pos - 1):
                cmp, branch = insns[k:k + 2]
                if (cmp.mnemonic == 'cmp' and len(cmp.operands) == 2
                        and cmp.operands[0].type == x86.X86_OP_REG and cmp.operands[1].type == x86.X86_OP_IMM
                        and family(cmp.operands[0].reg) == family(memory.index)
                        and branch.mnemonic == 'ja' and 0 <= cmp.operands[1].imm < 4096):
                    after_guard = insns[k + 2:load_pos]
                    if not any(i.group(cs.CS_GRP_CALL) or i.group(cs.CS_GRP_JUMP) or
                               any(family(reg) == family(memory.index) for reg in i.regs_access()[1]) for i in after_guard):
                        guards.append((cmp, branch))
            if not guards:
                continue
            cmp, branch = guards[-1]
            count = cmp.operands[1].imm + 1
            raw = read_memory(table, count * 4)
            if len(raw) != count * 4:
                continue
            targets = [table + delta[0] for delta in struct.iter_unpack('<i', raw)]
            if not all(a in addresses for a in targets):
                continue
            # The guard must dominate the dispatch under the known direct edges.
            direct = analyze_bytes(code, entry)
            jump_block = next(b['start'] for b in direct['blocks'] if int(b['start'], 16) <= jump.address <= int(b['end'], 16))
            guard_block = next(b['start'] for b in direct['blocks'] if int(b['start'], 16) <= branch.address <= int(b['end'], 16))
            node = jump_block
            dom = direct['immediate_dominators']
            while node in dom and node != guard_block and dom[node] != node:
                node = dom[node]
            if node != guard_block:
                continue
            if cmp.operands[0].size < 8:
                # A 32-bit comparison alone does not bound the upper half of a 64-bit index.
                definitions = [i for i in insns if i.address < cmp.address and
                               any(family(r) == family(memory.index) for r in i.regs_access()[1])]
                if not definitions:
                    continue
                definition = definitions[-1]
                if (definition.mnemonic not in ('mov', 'movzx', 'lea', 'and', 'or', 'xor', 'add', 'sub', 'imul', 'shl', 'shr')
                        or not definition.operands or definition.operands[0].type != x86.X86_OP_REG
                        or definition.operands[0].size != 4
                        or family(definition.operands[0].reg) != family(memory.index)):
                    continue
                definition_block = next(b['start'] for b in direct['blocks'] if
                                        int(b['start'], 16) <= definition.address <= int(b['end'], 16))
                node = guard_block
                while node in dom and node != definition_block and dom[node] != node:
                    node = dom[node]
                if node != definition_block:
                    continue
            tables.append({'jump_address': hex(jump.address), 'table_address': hex(table), 'index_register': family(memory.index),
                           'guard_address': hex(branch.address), 'index_upper_bound': count - 1,
                           'encoding': 'table base plus signed little-endian i32; index scale 4',
                           'entries': [{'index': i, 'target': hex(a)} for i, a in enumerate(targets)],
                           'limitation': 'Bound proven only over known direct edges; unresolved incoming indirect edges may invalidate it'})
    return tables


def binary_facts(binary, function, directory):
    directory.mkdir(parents=True, exist_ok=True)
    symbols = elf_symbols(binary)
    info = symbols[function["name"]]
    entry, size = info["address"], info["size"]
    with binary.open("rb") as stream:
        elf = ELFFile(stream)
        section = next(s for s in elf.iter_sections() if s["sh_addr"] <= entry < s["sh_addr"] + s["sh_size"])
        offset = entry - section["sh_addr"]
        code = section.data()[offset:offset + size]
        sections = [(s['sh_addr'], s.data()) for s in elf.iter_sections()
                    if s['sh_flags'] & 2 and s['sh_type'] != 'SHT_NOBITS']

    def read_memory(address, count):
        for start, data in sections:
            if start <= address and address + count <= start + len(data):
                return data[address - start:address - start + count]
        return b''

    names = {v["address"]: k for k, v in symbols.items() if isinstance(v, dict)}
    tables = relative_jump_tables(code, entry, read_memory)
    facts = analyze_bytes(code, entry, names, {int(t['jump_address'], 16): [int(e['target'], 16) for e in t['entries']]
                                             for t in tables})
    facts['jump_tables'] = tables
    # RIP-relative references and explicit disassembly labels constrain aliases.
    referenced = {int(a["rip_target"], 16) for a in facts["memory_accesses"] if a["rip_target"]}
    text = function.get("symbolized_disassembly", "")
    facts["global_symbols"] = [{"elf_name": name, "size": obj["size"], "address": hex(obj["address"]),
                                "c_alias": "binary_global_" + hashlib.sha256(name.encode()).hexdigest()[:12],
                                "assembler_label": '__asm__(' + json.dumps(name) + ')'}
                               for name, obj in symbols.items() if isinstance(obj, dict) and obj["kind"] == "STT_OBJECT"
                               and (obj["address"] in referenced or '<' + name + '>' in text or '<' + name + '+' in text)]
    facts["provenance"] = {"binary": str(binary), "function": function["name"],
                           "symbol_bytes_sha256": hashlib.sha256(code).hexdigest(), "decoder": "capstone"}
    write_json(directory / "binary_facts.json", facts)
    return facts


def ir_facts(clang, ir, directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "accepted.ll"
    source.write_text(ir)
    normalized = directory / "normalized.ll"
    opt = str(Path(clang).with_name("opt"))
    transform = run([opt, "-S", "-passes=verify,mem2reg,simplifycfg,instnamer,verify", str(source), "-o", str(normalized)])
    write_json(directory / "normalize.json", transform)
    if transform["returncode"]:
        return {"available": False, "diagnostic": transform["stderr"], "original_ir_unchanged": True}
    analysis = run([opt, "-passes=print<domtree>,print<loops>", "-disable-output", str(normalized)])
    write_json(directory / "analysis.json", analysis)
    return {"available": True, "normalized_ir": normalized.read_text(),
            "dominator_and_loop_report": analysis["stderr"],
            "provenance": "LLVM 18 opt passes on accepted model IR; not original source IR",
            "original_ir_unchanged": True}
