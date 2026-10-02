# Ghidra headless script: export an ELF's call graph and selected C functions.
from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import ConsoleTaskMonitor

import __main__
import json
import os


def is_external(function):
    if function.isExternal():
        return True
    if function.isThunk():
        thunk_target = function.getThunkedFunction(True)
        if thunk_target is not None and thunk_target.isExternal():
            return True
    block = currentProgram.getMemory().getBlock(function.getEntryPoint())
    return block is not None and (block.getName().startswith(".plt") or block.getName() == "EXTERNAL")


def function_record(function, monitor):
    calls = []
    for callee in function.getCalledFunctions(monitor):
        calls.append({
            "name": callee.getName(),
            "entry": str(callee.getEntryPoint()),
            "external": is_external(callee),
        })

    instructions = []
    instruction_records = []
    listing = currentProgram.getListing()
    for instruction in listing.getInstructions(function.getBody(), True):
        instructions.append(str(instruction))
        references = []
        for reference in instruction.getReferencesFrom():
            target = reference.getToAddress()
            symbol = currentProgram.getSymbolTable().getPrimarySymbol(target)
            data = listing.getDefinedDataAt(target)
            references.append({
                "address": str(target), "type": str(reference.getReferenceType()),
                "symbol": symbol.getName() if symbol is not None else None,
                "data": str(data.getDefaultValueRepresentation()) if data is not None else None,
            })
        instruction_records.append({"address": str(instruction.getAddress()),
                                    "instruction": str(instruction), "references": references})

    block = currentProgram.getMemory().getBlock(function.getEntryPoint())
    cfg = []
    blocks = BasicBlockModel(currentProgram).getCodeBlocksContaining(function.getBody(), monitor)
    while blocks.hasNext():
        code_block = blocks.next()
        successors = []
        destinations = code_block.getDestinations(monitor)
        while destinations.hasNext():
            destination = destinations.next()
            successors.append({"address": str(destination.getDestinationAddress()),
                               "flow_type": str(destination.getFlowType())})
        cfg.append({"start": str(code_block.getFirstStartAddress()),
                    "end": str(code_block.getMaxAddress()), "successors": successors})
    return {
        "name": function.getName(),
        "entry": str(function.getEntryPoint()),
        "external": is_external(function),
        "memory_block": block.getName() if block is not None else "",
        "signature": str(function.getSignature()),
        "assembly": "\n".join(instructions),
        "instruction_records": instruction_records,
        "cfg": cfg,
        "calls": calls,
    }


def decompile(function, interface, monitor):
    result = interface.decompileFunction(function, 60, monitor)
    if result.decompileCompleted():
        return result.getDecompiledFunction().getC()
    return None


args = __main__.getScriptArgs()
if len(args) < 1:
    raise ValueError("Expected output JSON path")

output_path = args[0]
requested_names = set()
if len(args) > 1 and args[1]:
    requested_names = set(args[1].split(","))

monitor = ConsoleTaskMonitor()
function_manager = currentProgram.getFunctionManager()
functions = list(function_manager.getFunctions(True))
records = []
selected = {}
for function in functions:
    if is_external(function):
        continue
    record = function_record(function, monitor)
    records.append(record)
    if function.getName() in requested_names:
        selected[function.getName()] = function

decompiled = {}
if selected:
    interface = DecompInterface()
    interface.openProgram(currentProgram)
    for name, function in selected.items():
        c_code = decompile(function, interface, monitor)
        if c_code is not None:
            decompiled[name] = c_code
    interface.dispose()

parent = os.path.dirname(output_path)
if parent and not os.path.exists(parent):
    os.makedirs(parent)
with open(output_path, "w") as output:
    json.dump({"schema_version": 2, "functions": records, "decompiled": decompiled}, output, indent=2)
