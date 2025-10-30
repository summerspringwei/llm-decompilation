import os
import re
import pickle
import tempfile
import subprocess
from typing import Optional, Tuple, Dict
from datasets import load_from_disk
from multiprocessing import Pool
from tqdm import tqdm

from exebench import Wrapper, LLVMAssembler

_DEFAULT_CMD_TIMEOUT = 90 # Increased timeout just in case Ghidra takes time to load

def _run_command(
        command, # command is a list of strings, not a single string
        stdin: Optional[str] = None,
        timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT) -> Tuple[int, str, str]:
    # Note: When passing a list of args, shell=False is the default and recommended
    output = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            input=stdin,
                            timeout=timeout)
    # No need to decode if text=True
    stdout = output.stdout
    stderr = output.stderr
    return output.returncode, stdout, stderr


# Try to get ghidra_script_path from environment variable first
ghidra_home = os.environ.get("GHIDRA_HOME")
if ghidra_home:
    ghidra_script_path = os.path.join(ghidra_home, "support/analyzeHeadless")
else:
    # Fall back to hardcoded path if env var not set
    ghidra_script_path = "/data1/xiachunwei/Software/ghidra_11.4.2_PUBLIC/support/analyzeHeadless"


class GhidraResult:
    def __init__(self):
        self.ghidra_decompiled_raw_string = None
        self.ghidra_decompiled_error_msg = None
        self.ghidra_decompiled_error_code = None
        self.ghidra_decompiled_c_code = None


class GhidraObjectFileDecompiler:
    def __init__(self, record):
        self.idx = -1
        self.record = record
        self.object_file_path = None
        self.synth_wrapper = None
        self.func_name = record["func_info"]["functions"][0]["name"]
        self.func_def = record["func_def"]
        self.ghidra_result = None
        
    def compile_to_object_file(self, output_dir: str, name_hint: str):
        """compile the llvm ir to object file"""
        object_file_path = f"{output_dir}/{name_hint}.o"
        ll_file_path = f"{output_dir}/{name_hint}.ll"
        with open(ll_file_path, "w") as f:
            f.write(self.record["llvm_ir"]["code"][-1])

        cmd = ["llc", ll_file_path, "-filetype=obj", "-o", object_file_path]
        retcode, stdout, stderr = _run_command(cmd, timeout=30)
        if retcode != 0:
            raise RuntimeError(f"Failed to compile the function to object file: {stderr}")
        return object_file_path

    def compile_to_executable(self, record) -> str: # Pass record as an argument
        """compile the object file to executable"""
        try:
            c_deps=(self.record['synth_deps'] + '\n' +
                    self.record['synth_io_pairs']['dummy_funcs'][0] + '\n').replace(
                        'typedef int bool;', '')
            self.synth_wrapper = Wrapper(
                c_deps=c_deps + '\n',
                func_c_signature=self.record['func_head_types'].replace('extern', ''),
                func_assembly=record["asm"]["code"][-1],
                cpp_wrapper=record['synth_exe_wrapper'],
                assembler_backend=LLVMAssembler())
        except Exception as e:
            raise RuntimeError(f"Failed to compile the function to executable: {e}")
        return self.synth_wrapper._compiled_exe_path


    def decompile_external_assembly(self, assembly: str, func_name: str) -> str:
        """decompile the external assembly"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            object_file_path = f"{tmp_dir}/{func_name}.o"
            assembly_file_path = f"{tmp_dir}/{func_name}.s"
            with open(assembly_file_path, "w") as f:
                f.write(assembly)
            cmd = ["clang", assembly_file_path, "-o", object_file_path]
            retcode, stdout, stderr = _run_command(cmd, timeout=30)
            if retcode != 0:
                raise RuntimeError(f"Failed to compile the assembly to object file: {stderr}")
            project_name = f"project_{self.idx}_predict_{func_name}"
            script_path = "models/ghidra_decompile/ghidra_decompile_script.py"
            cmd = [
                ghidra_script_path, 
                tmp_dir, 
                project_name, 
                "-import", object_file_path,  
                "-overwrite", # This will delete the project each time, which is good for loops
                "-postscript", script_path, 
                func_name
            ]
            retcode, stdout, stderr = _run_command(cmd, timeout=_DEFAULT_CMD_TIMEOUT)
            if retcode != 0:
                raise RuntimeError(f"Failed to run ghidra: {stderr}")
            else:
                return self.extract_c_code(stdout)


    def extract_c_code(self, stdout: str) -> str:
        """Extract C code between ```C and ``` markers from Ghidra output"""
        # Look for C code between ```C and ``` markers
        match = re.search(r'```C\n(.*?)```', stdout, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


def ghdria_decompile_record(args):
    """Process a single record with decompilation"""
    idx, record = args
    
    try:
        decompiler = GhidraObjectFileDecompiler(record)
        decompiler.idx = idx
        with tempfile.TemporaryDirectory() as tmp_dir:
            object_file_path = decompiler.compile_to_object_file(tmp_dir, decompiler.func_name)
            # It's better to create a unique project directory for each run
            project_dir = f"{tmp_dir}/myghidra_{idx}_{decompiler.func_name}" 
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)
            project_name = f"project_{idx}_{decompiler.func_name}"
            script_path = "models/ghidra_decompile/ghidra_decompile_script.py"

            # --- THIS IS THE CORRECTED LINE ---
            # 1. Removed the incorrect "-h" flag.
            # 2. Added "--" to separate Ghidra's arguments from your script's arguments.
            cmd = [
                ghidra_script_path, 
                project_dir, 
                project_name, 
                "-import", object_file_path,  
                "-overwrite", # This will delete the project each time, which is good for loops
                "-postscript", script_path, 
                decompiler.func_name
            ]
            retcode, stdout, stderr = _run_command(cmd, timeout=_DEFAULT_CMD_TIMEOUT)
            if retcode != 0:
                # Check if the error is just the "project exists" error, which can sometimes happen
                if "project exists" not in stderr:
                     raise RuntimeError(f"Failed to run ghidra: {stderr}")
            else:
                ghidra_result = GhidraResult()
                ghidra_result.ghidra_decompiled_raw_string = stdout
                ghidra_result.ghidra_decompiled_error_msg = stderr
                ghidra_result.ghidra_decompiled_error_code = retcode
                ghidra_result.ghidra_decompiled_c_code = decompiler.extract_c_code(stdout)
                decompiler.ghidra_result = ghidra_result
                
            return decompiler
    except Exception as e:
        print(f"Error processing record {idx}: {e}")
        return None


def main(dataset_path: str, num_processes: int = 4):
    dataset = load_from_disk(dataset_path)
    # Create a list of (idx, record) tuples for processing
    records_with_idx = list(enumerate(dataset))
    records_with_idx = [(idx, record) for idx, record in records_with_idx]
    with Pool(processes=num_processes) as pool:
        # Process records in parallel with progress bar
        results = list(tqdm(
            pool.imap(ghdria_decompile_record, records_with_idx),
            total=len(records_with_idx),
            desc="Processing records",
            unit="record"
        ))
    
    return results


if __name__ == "__main__":
    
    dataset_path_file_tyypes = {
        ("/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164", "sampled_dataset_without_loops_164_ghidra_decompile"),
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_164", "sampled_dataset_with_loops_164_ghidra_decompile"),
        # ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164", "sampled_dataset_with_loops_and_only_one_bb_164_ghidra_decompile"),
    }
    for dataset_path, file_name in dataset_path_file_tyypes:
        results = main(dataset_path, num_processes=40)
        pickle.dump(results, open(f"{file_name}.pkl", "wb"))