#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.pcode import PcodeOp


def _find_function_by_name(func_name, program):
    funcs = getGlobalFunctions(func_name)
    if funcs and len(funcs) > 0:
        return funcs[0]
    func_manager = program.getFunctionManager()
    for func in func_manager.getFunctions(True):
        if func.getName() == func_name:
            return func
    return None


def _build_addr_to_bb(high_func):
    """Build mapping from block start address to BB label (BB0, BB1, ...)."""
    blocks = list(high_func.getBasicBlocks())
    # Sort by start address so entry block is first and order is deterministic
    blocks.sort(key=lambda b: b.getStart().getOffset())
    addr_to_bb = {}
    for i, block in enumerate(blocks):
        addr_to_bb[block.getStart()] = "BB{}".format(i)
    return addr_to_bb


def _get_symbol_at_address(addr, symbol_table):
    """Return the primary symbol name at this address (e.g. function name), or None."""
    if symbol_table is None:
        return None
    symbol = symbol_table.getPrimarySymbol(addr)
    if symbol is not None:
        name = symbol.getName()
        if name is not None and name != "":
            return name
    return None


def _format_varnode(varnode, language, symbol_table, addr_to_bb):
    """Format a Varnode with human-readable register names; block starts as BB0, BB1, ..."""
    if varnode.isRegister():
        reg_name = varnode.toString(language)
        return "(register, {}, {})".format(reg_name, varnode.getSize())
    elif varnode.isConstant():
        return "(const, 0x{:x}, {})".format(varnode.getOffset(), varnode.getSize())
    elif varnode.isUnique():
        return "(unique, 0x{:x}, {})".format(varnode.getOffset(), varnode.getSize())
    elif varnode.isAddress():
        addr = varnode.getAddress()
        # Use BB label if this address is the start of a basic block (e.g. jump target)
        if addr_to_bb is not None and addr in addr_to_bb:
            return "(ram, {}, {})".format(addr_to_bb[addr], varnode.getSize())
        # Prefer symbol/function name (e.g. for CALL target)
        symbol_name = _get_symbol_at_address(addr, symbol_table)
        if symbol_name is not None:
            return "(ram, {}, {})".format(symbol_name, varnode.getSize())
        # Fallback: hex address
        return "(ram, 0x{:x}, {})".format(addr.getOffset(), varnode.getSize())
    else:
        return varnode.toString()


def _format_pcode_op(pcode_op, language, symbol_table, addr_to_bb, include_addr_and_id=False):
    """Format a P-code operation. No address/instruction id unless include_addr_and_id."""
    opcode = pcode_op.getOpcode()
    try:
        opcode_name = pcode_op.getMnemonic()
    except:
        opcode_name = str(opcode)

    output = pcode_op.getOutput()
    output_str = ""
    if output is not None:
        output_str = _format_varnode(output, language, symbol_table, addr_to_bb) + " = "

    inputs = []
    for i in range(pcode_op.getNumInputs()):
        input_varnode = pcode_op.getInput(i)
        formatted_input = _format_varnode(input_varnode, language, symbol_table, addr_to_bb)
        inputs.append(formatted_input)

    CBRANCH = PcodeOp.CBRANCH
    BRANCH = PcodeOp.BRANCH
    CALL = PcodeOp.CALL
    CALLIND = PcodeOp.CALLIND
    RETURN = PcodeOp.RETURN
    STORE = PcodeOp.STORE

    if opcode == CBRANCH:
        op_str = " ---  CBRANCH " + " , ".join(inputs)
    elif opcode == BRANCH:
        op_str = " ---  BRANCH " + " , ".join(inputs)
    elif opcode == CALL or opcode == CALLIND:
        op_str = " ---  " + opcode_name + " " + " , ".join(inputs)
    elif opcode == RETURN:
        op_str = " ---  RETURN " + " , ".join(inputs) if inputs else " ---  RETURN"
    elif opcode == STORE:
        op_str = " ---  STORE " + " , ".join(inputs)
    else:
        if output_str:
            op_str = output_str + opcode_name + " " + " , ".join(inputs)
        else:
            op_str = opcode_name + " " + " , ".join(inputs)

    if include_addr_and_id:
        seq = pcode_op.getSeqnum()
        return "[{}:{}] {}".format(seq.getTarget().toString(), seq.getTime(), op_str)
    return op_str


def decompile_to_pcode_text(func_name, current_program):
    func = _find_function_by_name(func_name, current_program)
    if func is None:
        raise ValueError("Function {} not found".format(func_name))
    ifc = DecompInterface()
    ifc.openProgram(current_program)
    results = ifc.decompileFunction(func, 60, ConsoleTaskMonitor())
    if not results.decompileCompleted():
        raise RuntimeError("Decompilation failed.")
    high_func = results.getHighFunction()

    language = current_program.getLanguage()
    symbol_table = current_program.getSymbolTable()
    addr_to_bb = _build_addr_to_bb(high_func)

    lines = []
    lines.append("function {}".format(func.getName()))
    entry_addr = func.getEntryPoint()
    entry_bb = addr_to_bb.get(entry_addr)
    if entry_bb is None:
        # Entry might match a block start; if not, use BB0
        blocks = list(high_func.getBasicBlocks())
        blocks.sort(key=lambda b: b.getStart().getOffset())
        entry_bb = addr_to_bb[blocks[0].getStart()]
    lines.append("entry {}".format(entry_bb))

    blocks = list(high_func.getBasicBlocks())
    blocks.sort(key=lambda b: b.getStart().getOffset())
    for block in blocks:
        lines.append("")
        bb_label = addr_to_bb[block.getStart()]
        lines.append("## block {}".format(bb_label))
        op_iter = block.getIterator()
        while op_iter.hasNext():
            lines.append(_format_pcode_op(op_iter.next(), language, symbol_table, addr_to_bb, include_addr_and_id=False))
    return "\n".join(lines)


import __main__
args = __main__.getScriptArgs()
if len(args) == 0:
    raise ValueError("No function name provided")
func_name = args[0]
output_file = args[1] if len(args) > 1 else None

pcode_text = decompile_to_pcode_text(func_name, __main__.currentProgram)
if output_file:
    with open(output_file, "w") as f:
        f.write(pcode_text)
else:
    print(pcode_text)
