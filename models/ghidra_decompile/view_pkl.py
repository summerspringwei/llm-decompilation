import pickle
from models.ghidra_decompile.get_object_file import ObjectFileDecompiler
from models.ghidra_decompile.get_object_file import GhidraResult
results = pickle.load(open("sampled_dataset_without_loops_164_ghidra_decompile.pkl", "rb"))
for result in results:
    print(result.func_name)
    print("xx")
    print(result.ghidra_result.ghidra_decompiled_raw_string)
    print("-"*100)
    print(result.record["func_def"])
    print("="*100)