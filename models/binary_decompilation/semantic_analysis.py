"""Conservative raw P-code dataflow. No source types or decompiled bodies are read."""

import hashlib
import json
from pathlib import Path

from elftools.elf.elffile import ELFFile

from models.binary_decompilation.compiler_evidence import binary_facts
from models.binary_decompilation.native_validation import elf_symbols


VERSION = 1
ARGUMENTS = ("RDI", "RSI", "RDX", "RCX", "R8", "R9")
CLOBBERS = ("RAX", "RCX", "RDX", "RSI", "RDI", "R8", "R9", "R10", "R11",
            "CF", "PF", "AF", "ZF", "SF", "OF") + tuple("XMM%d" % n for n in range(16))


def function_bytes(binary, name):
    symbol = elf_symbols(binary)[name]
    with Path(binary).open("rb") as stream:
        elf = ELFFile(stream)
        if elf.get_machine_arch() != "x64":
            raise ValueError("Semantic analysis currently supports x86-64 ELF only")
        for section in elf.iter_sections():
            offset = symbol["address"] - section["sh_addr"]
            if 0 <= offset and offset + symbol["size"] <= section["sh_size"]:
                return section.data()[offset:offset + symbol["size"]], symbol["address"]
    raise ValueError("Function bytes are not contained in one ELF section")


def lift(code, entry):
    import capstone
    import pypcode

    context = pypcode.Context("x86:LE:64:default")
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    records, errors = [], []

    def node(var, address):
        result = {"space": var.space.name, "offset": var.offset, "size": var.size}
        if var.space.name == "register":
            result["name"] = context.getRegisterName(var.space, var.offset, var.size)
        if var.space.name == "unique":
            result["scope"] = address
        return result

    for ins in decoder.disasm(code, entry):
        address = hex(ins.address)
        try:
            ops = context.translate(bytes(ins.bytes), ins.address, max_instructions=1).ops
            for index, op in enumerate(ops):
                opcode = op.opcode.name
                if opcode == "IMARK":
                    continue
                # LOAD/STORE input zero is an opaque process-local space pointer.
                inputs = list(op.inputs)[1:] if opcode in ("LOAD", "STORE") else op.inputs
                records.append({"id": address + ":" + str(index), "address": address,
                                "opcode": opcode, "inputs": [node(v, address) for v in inputs],
                                "output": node(op.output, address) if op.output else None})
        except (pypcode.BadDataError, pypcode.UnimplError) as error:
            errors.append({"address": address, "reason": str(error)})
    if sum(i.size for i in decoder.disasm(code, entry)) != len(code):
        errors.append({"address": hex(entry), "reason": "Incomplete instruction decoding"})
    registers = {name: node(context.registers[name], "entry") for name in (*ARGUMENTS, *CLOBBERS, "RSP")
                 if name in context.registers}
    return records, errors, registers


def cells(node):
    if not node or node["space"] not in ("register", "unique"):
        return []
    prefix = node["space"] + ":" + node.get("scope", "") + ":"
    return [prefix + str(node["offset"] + i) for i in range(node["size"])]


def constant(node, known):
    if node["space"] == "const":
        return node["offset"]
    values = [known.get(c) for c in cells(node)]
    if values and all(v is not None for v in values):
        return sum(v << (8 * i) for i, v in enumerate(values))
    return None


def bit_facts(node, known):
    width = node['size'] * 8
    mask = (1 << width) - 1
    if node['space'] == 'const':
        value, known_mask = node['offset'] & mask, mask
    else:
        value = sum(known[c] << (8 * i) for i, c in enumerate(cells(node)) if c in known)
        known_mask = sum(255 << (8 * i) for i, c in enumerate(cells(node)) if c in known)
    return {'width_bits': width, 'known_mask': hex(known_mask), 'known_value': hex(value),
            'unsigned_min': value, 'unsigned_max': value | (mask ^ known_mask)}


def evaluate(opcode, args, sizes, width):
    if any(a is None for a in args):
        return None
    mask = (1 << (8 * width)) - 1
    a = args[0] if args else 0
    b = args[1] if len(args) > 1 else 0
    signed = lambda v, size: v - (1 << (8 * size)) if v & (1 << (8 * size - 1)) else v
    operations = {
        "COPY": lambda: a, "INT_ZEXT": lambda: a,
        "INT_SEXT": lambda: signed(a, sizes[0]),
        "INT_ADD": lambda: a + b, "INT_SUB": lambda: a - b,
        "INT_MULT": lambda: a * b, "INT_AND": lambda: a & b,
        "INT_OR": lambda: a | b, "INT_XOR": lambda: a ^ b,
        "INT_NEGATE": lambda: ~a, "INT_2COMP": lambda: -a,
        "INT_LEFT": lambda: a << b if b < width * 8 else 0,
        "INT_RIGHT": lambda: a >> b if b < sizes[0] * 8 else 0,
        "INT_SRIGHT": lambda: signed(a, sizes[0]) >> min(b, sizes[0] * 8),
        "INT_EQUAL": lambda: int(a == b), "INT_NOTEQUAL": lambda: int(a != b),
        "INT_LESS": lambda: int(a < b), "INT_LESSEQUAL": lambda: int(a <= b),
        "INT_SLESS": lambda: int(signed(a, sizes[0]) < signed(b, sizes[1])),
        "INT_SLESSEQUAL": lambda: int(signed(a, sizes[0]) <= signed(b, sizes[1])),
        "BOOL_NEGATE": lambda: int(not a), "BOOL_AND": lambda: int(bool(a) and bool(b)),
        "BOOL_OR": lambda: int(bool(a) or bool(b)), "BOOL_XOR": lambda: int(bool(a) != bool(b)),
        "PIECE": lambda: (a << (sizes[1] * 8)) | b, "SUBPIECE": lambda: a >> (b * 8),
        "POPCOUNT": lambda: a.bit_count(),
    }
    return operations[opcode]() & mask if opcode in operations else None


def backward_slice(roots, dependencies, limit=200):
    pending, result = list(roots), set()
    while pending and len(result) < limit:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(dependencies.get(current, []))
    return {"definitions": sorted(result), "truncated": bool(pending)}


def affine_memory(records, cfg, registers, max_iterations=64):
    """Propagate exact symbolic-base + byte-offset expressions, widening at joins."""
    def key(node):
        return tuple(cells(node))

    initial = {key(registers[name]): ("entry:" + name, 0) for name in (*ARGUMENTS, "RSP")}
    preds = {b['start']: [] for b in cfg['blocks']}
    for b in cfg['blocks']:
        for dest in b['successors']:
            if dest in preds:
                preds[dest].append(b['start'])
    outputs, accesses = {}, {}
    converged = False
    for iteration in range(max_iterations):
        changed = False
        for block in cfg['blocks']:
            incoming = [outputs[p] for p in preds[block['start']] if p in outputs]
            if block['start'] == cfg['entry']:
                incoming.append(initial)
            if cfg.get('unresolved_indirect_jumps'):
                incoming.append({})
            if not incoming:
                continue
            state = {k: v for k, v in incoming[0].items() if all(s.get(k) == v for s in incoming)}
            block_accesses, uncertain = [], False
            for r in records:
                if not int(block['start'], 16) <= int(r['address'], 16) <= int(block['end'], 16):
                    continue
                ins, out, opcode = r['inputs'], r['output'], r['opcode']
                value = lambda n: ("constant", n['offset']) if n['space'] == 'const' else state.get(key(n))
                args = [value(n) for n in ins]
                result = None
                if opcode in ('LOAD', 'STORE'):
                    address = args[0]
                    accesses[r['id']] = {'instruction': r['address'], 'operation': opcode,
                        'width_bytes': out['size'] if opcode == 'LOAD' else ins[-1]['size'],
                        'address_expression': {'base': address[0], 'offset': address[1]} if address else None,
                        'status': 'conditional' if address else 'unknown'}
                    block_accesses.append(r['id'])
                if opcode == 'COPY' and args:
                    result = args[0]
                elif opcode in ('INT_ADD', 'INT_SUB') and all(a is not None for a in args):
                    a, b = args
                    if b[0] == 'constant':
                        result = (a[0], a[1] + (b[1] if opcode == 'INT_ADD' else -b[1]))
                    elif opcode == 'INT_ADD' and a[0] == 'constant':
                        result = (b[0], a[1] + b[1])
                if out:
                    overlap = set(cells(out))
                    state = {k: v for k, v in state.items() if not overlap.intersection(k)}
                    if result and (out['size'] == 8 or result[0] == 'constant'):
                        bits = out['size'] * 8
                        offset = result[1] % (1 << bits)
                        if result[0] != 'constant' and offset >= 1 << (bits - 1):
                            offset -= 1 << bits
                        state[key(out)] = (result[0], offset)
                if opcode in ('CALL', 'CALLIND'):
                    killed = {c for name in CLOBBERS if name in registers for c in cells(registers[name])}
                    state = {k: v for k, v in state.items() if not killed.intersection(k)}
                    sp = key(registers['RSP'])
                    if sp in state:
                        state[sp] = (state[sp][0], state[sp][1] + 8)
                if opcode == 'CALLOTHER' or opcode in ('CBRANCH', 'BRANCH') and ins[0]['space'] == 'const':
                    state = {}
                    uncertain = True
            if uncertain:
                state = {}
                for access_id in block_accesses:
                    accesses[access_id].update(status='unknown', address_expression=None)
            state = {k: v for k, v in state.items() if k and k[0].startswith('register:')}
            if outputs.get(block['start']) != state:
                outputs[block['start']] = state
                changed = True
        if not changed:
            converged = True
            break
    if not converged:
        for access in accesses.values():
            access.update(status='unknown', address_expression=None)
    layouts = {}
    for access in accesses.values():
        address = access['address_expression']
        if address:
            layouts.setdefault(address['base'], set()).add((address['offset'], access['width_bytes'], access['operation']))
    return {'converged': converged, 'accesses': list(accesses.values()),
            'layout_constraints': {base: [{'offset': o, 'width_bytes': w, 'access': a} for o, w, a in sorted(fields)]
                                   for base, fields in layouts.items()},
            'limitations': ['Affine equality only; differing join values widen to unknown',
                            'Argument-relative addresses do not establish pointer validity or non-aliasing']}


def dataflow(records, cfg, registers, max_iterations=64):
    """Byte-level may-definitions plus must-constants; uncertain control is a barrier."""
    blocks = cfg["blocks"]
    by_block = {b["start"]: [r for r in records if int(b["start"], 16) <= int(r["address"], 16)
                              <= int(b["end"], 16)] for b in blocks}
    predecessors = {b["start"]: [] for b in blocks}
    for b in blocks:
        for dest in b["successors"]:
            if dest in predecessors:
                predecessors[dest].append(b["start"])
    outputs, dependencies, observations, call_abi = {}, {}, {}, {}
    barriers = []
    all_cells = {c for r in records for n in [*r["inputs"], r["output"]] for c in cells(n)}
    for n in registers.values():
        all_cells.update(cells(n))
    initial = {c: frozenset(["entry:" + c]) for c in all_cells if c.startswith("register:")}
    uncertain_graph = bool(cfg.get("unresolved_indirect_jumps"))

    def transfer(ops, definitions, known):
        definitions, known = dict(definitions), dict(known)
        for r in ops:
            opcode, ins, out = r["opcode"], r["inputs"], r["output"]
            used = {d for n in ins for c in cells(n) for d in definitions.get(c, ["entry:" + c])}
            implicit = ARGUMENTS if opcode in ("CALL", "CALLIND") else ("RAX",) if opcode == "RETURN" else ()
            used.update(d for name in implicit for c in cells(registers[name])
                        for d in definitions.get(c, ["entry:" + c]))
            dependencies[r["id"]] = sorted(used)
            values = [constant(n, known) for n in ins]
            observations[r["id"]] = {"inputs": values, 'known_bits': [bit_facts(n, known) for n in ins]}
            if opcode in ('CALL', 'CALLIND'):
                call_abi[r['id']] = {'instruction': r['address'], 'status': 'possible_machine_arguments',
                                    'registers': {name: {'bits': bit_facts(registers[name], known),
                                     'definitions': sorted({d for c in cells(registers[name])
                                                           for d in definitions.get(c, ['entry:' + c])})}
                                                  for name in ARGUMENTS}}
            internal_branch = opcode in ("BRANCH", "CBRANCH") and ins and ins[0]["space"] == "const"
            if internal_branch or opcode in ("CALLOTHER", "CALL", "CALLIND"):
                targets = registers.values() if internal_branch or opcode == "CALLOTHER" else (
                    registers[n] for n in CLOBBERS if n in registers)
                for target in targets:
                    for c in cells(target):
                        definitions[c] = frozenset([r["id"]])
                        known.pop(c, None)
                if internal_branch:
                    # Later micro-ops are path-dependent; invalidate the entire block below.
                    barriers.append(r["id"])
            if out:
                value = evaluate(opcode, values, [n["size"] for n in ins], out["size"])
                for i, c in enumerate(cells(out)):
                    definitions[c] = frozenset([r["id"]])
                    known.pop(c, None)
                    if value is not None:
                        known[c] = (value >> (8 * i)) & 255
                # A zero extension has known high bytes even when its operand is unknown.
                if opcode == "INT_ZEXT":
                    for c in cells(out)[ins[0]["size"]:]:
                        known[c] = 0
        if any(r["id"] in barriers for r in ops):
            definitions = {c: frozenset(["unknown:internal_pcode_control"]) for c in all_cells}
            known = {}
        return ({c: ds for c, ds in definitions.items() if c.startswith("register:")},
                {c: v for c, v in known.items() if c.startswith("register:")})

    converged = False
    for iteration in range(max_iterations):
        changed = False
        for b in blocks:
            start = b["start"]
            incoming = [outputs[p] for p in predecessors[start] if p in outputs]
            if start == cfg["entry"] or uncertain_graph:
                incoming.append((initial, {}))
            if not incoming:
                continue
            definitions = {}
            for c in set().union(*(set(d) for d, _ in incoming)):
                values = frozenset().union(*(d.get(c, frozenset(["entry:" + c])) for d, _ in incoming))
                definitions[c] = values if len(values) <= 32 else frozenset(["unknown:definition_limit"])
            first_known = incoming[0][1]
            known = {c: v for c, v in first_known.items() if all(k.get(c) == v for _, k in incoming)}
            result = transfer(by_block[start], definitions, known)
            if outputs.get(start) != result:
                changed = True
                outputs[start] = result
        if not changed:
            converged = True
            break
    interfaces = []
    for b in blocks:
        ops = by_block[b["start"]]
        reads, writes = set(), set()
        for r in ops:
            reads.update(c for n in r["inputs"] for c in cells(n) if c.startswith("register:") and c not in writes)
            implicit = ARGUMENTS if r['opcode'] in ('CALL', 'CALLIND') else ('RAX',) if r['opcode'] == 'RETURN' else ()
            reads.update(c for name in implicit for c in cells(registers[name]) if c not in writes)
            writes.update(c for c in cells(r["output"]) if c.startswith("register:"))
            if r['opcode'] in ('CALL', 'CALLIND'):
                writes.update(c for name in CLOBBERS if name in registers for c in cells(registers[name]))
        interfaces.append({"block": b["start"], "successors": b["successors"],
                           "read_before_write_bytes": sorted(reads), "written_bytes": sorted(writes)})
    live_in, live_out = {b["block"]: set() for b in interfaces}, {b["block"]: set() for b in interfaces}
    liveness_converged = False
    for _ in range(max_iterations):
        changed = False
        for b in reversed(interfaces):
            outgoing = set().union(*(live_in.get(s, set()) for s in b["successors"]))
            incoming = set(b["read_before_write_bytes"]) | (outgoing - set(b["written_bytes"]))
            if outgoing != live_out[b["block"]] or incoming != live_in[b["block"]]:
                changed = True
            live_in[b["block"]], live_out[b["block"]] = incoming, outgoing
        if not changed:
            liveness_converged = True
            break
    for b in interfaces:
        b["live_in_bytes"], b["live_out_bytes"] = sorted(live_in[b["block"]]), sorted(live_out[b["block"]])
    roots = [r["id"] for r in records if r["opcode"] in ("STORE", "CBRANCH", "CALL", "CALLIND", "RETURN")]
    return {"converged": converged, "liveness_converged": liveness_converged, "iterations": iteration + 1,
            "status": "conditional" if converged and liveness_converged and not barriers and not uncertain_graph else "unknown",
            "assumptions": ["Known CFG edges are complete", "SysV AMD64 calling convention", "P-code semantics are accurate"],
            "barriers": sorted(set(barriers)), "block_interfaces": interfaces,
            "dependencies": dependencies, "constant_observations": observations,
            "call_abi_constraints": list(call_abi.values()),
            "observable_slices": {root: backward_slice([root], dependencies) for root in roots},
            "entry_argument_registers_read": [name for name in ARGUMENTS if any(
                "entry:" + c in ds for c in cells(registers[name]) for ds in dependencies.values())],
            "limitations": ["Argument reads constrain machine inputs, not source parameter count or types",
                            "No memory value propagation or alias disambiguation",
                            "Unknown operations produce unknown values; internal P-code branches invalidate block facts",
                            "Unknown indirect CFG edges prevent completeness claims"]}


def analyze_function(binary, function, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import pypcode
    identity = {"version": VERSION, "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                "function": function["name"], "pypcode": pypcode.__version__,
                "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cfg_implementation": hashlib.sha256(Path(__file__).with_name("compiler_evidence.py").read_bytes()).hexdigest()}
    path = directory / "analysis_facts.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] == identity:
            return previous
    cfg = binary_facts(Path(binary), function, directory / "cfg")
    code, entry = function_bytes(binary, function["name"])
    records, errors, registers = lift(code, entry)
    flow = dataflow(records, cfg, registers)
    if errors:
        flow["status"] = "unknown"
    result = {"schema_version": VERSION, "identity": identity, "cfg": cfg,
              "pcode": records, "lift_errors": errors, "dataflow": flow,
              "memory": affine_memory(records, cfg, registers)}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def prompt_packet(facts):
    flow = facts["dataflow"]
    operations = facts["pcode"]
    # Full evidence is saved on disk; the prompt emphasizes observable slices.
    useful = set().union(*(set(s["definitions"]) for s in flow["observable_slices"].values()))
    selected = [{**r, 'abstract_inputs': flow['constant_observations'].get(r['id'], {})}
                for r in operations if r["id"] in useful][:600]
    def ranges(byte_cells):
        offsets = sorted(int(c.rsplit(':', 1)[1]) for c in byte_cells)
        result = []
        for offset in offsets:
            if result and result[-1][0] + result[-1][1] == offset:
                result[-1][1] += 1
            else:
                result.append([offset, 1])
        return result
    interfaces = [{key: ranges(value) if key.endswith('_bytes') else value for key, value in block.items()}
                  for block in flow['block_interfaces']]
    register_layout = {n['name']: [n['offset'], n['size']] for r in operations for n in [*r['inputs'], r['output']]
                       if n and n['space'] == 'register' and n.get('name')}
    # Full definitions are kept in the artifact; call packets expose widths/ranges, not repeated byte IDs.
    calls = [{**c, 'registers': {name: {'bits': item['bits'], 'definitions': item['definitions'][:8],
                                      'omitted_definitions': max(0, len(item['definitions']) - 8)}
                                for name, item in c['registers'].items()}}
             for c in flow['call_abi_constraints']]
    return {"schema_version": VERSION, "status": flow["status"], "assumptions": flow["assumptions"],
            "argument_register_evidence": flow["entry_argument_registers_read"],
            "block_interfaces": interfaces, "register_layout": register_layout,
            "byte_range_encoding": "[register-space byte offset, byte count]", "operation_semantics": selected,
            "included_operations": len(selected), "total_operations": len(operations),
            "memory_regions": facts['memory'],
            "call_abi_constraints": calls,
            "omitted_operations": len(operations) - len(selected),
            "limitations": flow["limitations"], "lift_errors": facts["lift_errors"]}
