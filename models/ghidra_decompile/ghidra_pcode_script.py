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


def _format_address_with_label(address, symbol_table):
    """Format an address with its label if available."""
    symbol = symbol_table.getPrimarySymbol(address)
    if symbol is not None and symbol.getName() != "":
        return symbol.getName()
    return address.toString()


def _format_varnode(varnode, language, symbol_table):
    """Format a Varnode with human-readable register names and labels."""
    if varnode.isRegister():
        # Use language-specific toString for register names
        # This returns the register name like "RAX", "RBX", etc.
        reg_name = varnode.toString(language)
        # Format as (register, <name>, <size>) to match original format
        return "(register, {}, {})".format(reg_name, varnode.getSize())
    elif varnode.isConstant():
        # Format constants
        return "(const, 0x{:x}, {})".format(varnode.getOffset(), varnode.getSize())
    elif varnode.isUnique():
        # Format unique temporaries
        return "(unique, 0x{:x}, {})".format(varnode.getOffset(), varnode.getSize())
    elif varnode.isAddress():
        # Format addresses with labels
        addr = varnode.getAddress()
        label = _format_address_with_label(addr, symbol_table)
        if label != addr.toString():
            # Use label if available
            return "(ram, {}, {})".format(label, varnode.getSize())
        else:
            # Fallback to hex address
            return "(ram, 0x{:x}, {})".format(addr.getOffset(), varnode.getSize())
    else:
        # Fallback to default representation
        return varnode.toString()


def _format_pcode_op(pcode_op, language, symbol_table):
    """Format a P-code operation with human-readable symbols."""
    seq = pcode_op.getSeqnum()
    addr = seq.getTarget()
    addr_str = _format_address_with_label(addr, symbol_table)
    
    # Get operation mnemonic
    opcode = pcode_op.getOpcode()
    try:
        opcode_name = pcode_op.getMnemonic()
    except:
        # Fallback if getMnemonic() doesn't exist
        opcode_name = str(opcode)
    
    # Format output (if any)
    output = pcode_op.getOutput()
    output_str = ""
    if output is not None:
        output_str = _format_varnode(output, language, symbol_table) + " = "
    
    # Format inputs
    inputs = []
    for i in range(pcode_op.getNumInputs()):
        input_varnode = pcode_op.getInput(i)
        formatted_input = _format_varnode(input_varnode, language, symbol_table)
        
        # Special handling: if this is an address input for control flow, try to get label
        if input_varnode.isAddress():
            addr_input = input_varnode.getAddress()
            label = _format_address_with_label(addr_input, symbol_table)
            if label != addr_input.toString():
                formatted_input = "(ram, {}, {})".format(label, input_varnode.getSize())
        
        inputs.append(formatted_input)
    
    # Build the operation string
    # PcodeOp opcode constants
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
    
    return "[{}:{}] {}".format(addr_str, seq.getTime(), op_str)


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
    
    # Get language and symbol table for formatting
    language = current_program.getLanguage()
    symbol_table = current_program.getSymbolTable()
    
    lines = []
    lines.append("function {}".format(func.getName()))
    entry_addr = func.getEntryPoint()
    entry_label = _format_address_with_label(entry_addr, symbol_table)
    lines.append("entry {}".format(entry_label))
    block_iter = high_func.getBasicBlocks()
    for block in block_iter:
        op_iter = block.getIterator()
        lines.append("")
        start_addr = block.getStart()
        stop_addr = block.getStop()
        start_label = _format_address_with_label(start_addr, symbol_table)
        stop_label = _format_address_with_label(stop_addr, symbol_table)
        lines.append("## block {} -> {}".format(start_label, stop_label))
        while op_iter.hasNext():
            lines.append(_format_pcode_op(op_iter.next(), language, symbol_table))
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
