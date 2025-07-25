# -*- coding: utf-8 -*-
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.decompiler import DecompInterface


def decompile_function(func_name, currentProgram):
    funcs = getGlobalFunctions(func_name)
    if len(funcs) == 0:
        raise ValueError("Function {} not found".format(func_name))

    func = funcs[0]
    ifc = DecompInterface()
    ifc.openProgram(currentProgram)
    results = ifc.decompileFunction(func, 60, ConsoleTaskMonitor())

    if not results.decompileCompleted():
        raise RuntimeError("Decompilation failed.")
    print("```C")
    c_func = results.getDecompiledFunction().getC()
    print(c_func)
    print("```")

    return c_func


import __main__
args = __main__.getScriptArgs()
if len(args) == 0:
    raise ValueError("No function name provided")
decompile_function(args[0], __main__.currentProgram)
