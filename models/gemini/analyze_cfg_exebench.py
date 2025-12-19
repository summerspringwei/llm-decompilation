import os
import pickle
import json
import subprocess
import tempfile
from typing import List
from datasets import load_from_disk
from models.gemini.llm_decompiler import LLMDecompileRecord


def analyze_cfg_exebench(validation_dir: str):
    validation_results: List[LLMDecompileRecord] = pickle.load(open(os.path.join(validation_dir, 'results.pkl'), 'rb'))
    for llm_decompile_record in validation_results:
        analyze_one_llm_decompile_record(llm_decompile_record)

def _run_command(command, stdin=None, timeout=90):
    """Run a command and return (returncode, stdout, stderr)."""
    output = subprocess.run(
        command,
        capture_output=True,
        text=True,
        input=stdin,
        timeout=timeout
    )
    return output.returncode, output.stdout, output.stderr


def _extract_cfg_from_object_file(object_file_path: str, func_name: str, output_json_path: str) -> dict:
    """Extract CFG JSON from an object file using ghidra_extract_bb.py."""
    ghidra_home = os.environ.get("GHIDRA_HOME")
    if ghidra_home:
        ghidra_script_path = os.path.join(ghidra_home, "support/analyzeHeadless")
    else:
        ghidra_script_path = "/data1/xiachunwei/Software/ghidra_11.4.2_PUBLIC/support/analyzeHeadless"
    
    script_path = os.path.join(os.path.dirname(__file__), "..", "ghidra_decompile", "ghidra_extract_bb.py")
    script_path = os.path.abspath(script_path)
    
    # Create a temporary directory for the Ghidra project
    with tempfile.TemporaryDirectory() as tmp_dir:
        project_name = f"project_{os.path.basename(object_file_path).replace('.o', '')}"
        
        cmd = [
            ghidra_script_path,
            tmp_dir,
            project_name,
            "-import", object_file_path,
            "-overwrite",
            "-postscript", script_path,
            func_name,
            output_json_path
        ]
        
        retcode, stdout, stderr = _run_command(cmd, timeout=120)
        if retcode != 0:
            print(f"Error running Ghidra: {stderr}")
            return None
        
        # Load the JSON file if it was created
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: JSON file {output_json_path} was not created")
            return None


def _compare_cfg_structures(predict_cfg: dict, target_cfg: dict, func_name: str) -> None:
    """Compare two CFG structures and print differences."""
    print(f"\n{'='*60}")
    print(f"CFG Comparison for function: {func_name}")
    print(f"{'='*60}\n")
    
    # Extract the function data from both CFGs
    # The structure is: {program_path: {arch: ..., func_addr: {nodes, edges, basic_blocks}}}
    predict_func_data = None
    target_func_data = None
    
    for prog_path, prog_data in predict_cfg.items():
        if isinstance(prog_data, dict) and "arch" in prog_data:
            for func_addr, func_data in prog_data.items():
                if func_addr != "arch" and isinstance(func_data, dict) and "nodes" in func_data:
                    predict_func_data = func_data
                    break
        if predict_func_data:
            break
    
    for prog_path, prog_data in target_cfg.items():
        if isinstance(prog_data, dict) and "arch" in prog_data:
            for func_addr, func_data in prog_data.items():
                if func_addr != "arch" and isinstance(func_data, dict) and "nodes" in func_data:
                    target_func_data = func_data
                    break
        if target_func_data:
            break
    
    if not predict_func_data:
        print("ERROR: Could not extract function data from predict CFG")
        return
    
    if not target_func_data:
        print("ERROR: Could not extract function data from target CFG")
        return
    
    predict_nodes = set(predict_func_data.get("nodes", []))
    target_nodes = set(target_func_data.get("nodes", []))
    
    predict_edges = set(tuple(edge) for edge in predict_func_data.get("edges", []))
    target_edges = set(tuple(edge) for edge in target_func_data.get("edges", []))
    
    # Use networkx to compare control flow graphs
    try:
        import networkx as nx
    except ImportError:
        print("networkx is not installed. Please install it to compare CFGs using graphs.")
        return

    def build_cfg_graph(nodes, edges):
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n)
        for src, dst in edges:
            G.add_edge(src, dst)
        return G

    # Build graphs
    G_predict = build_cfg_graph(predict_nodes, predict_edges)
    G_target = build_cfg_graph(target_nodes, target_edges)

    # Basic stats
    print(f"Predict CFG: {G_predict.number_of_nodes()} nodes, {G_predict.number_of_edges()} edges")
    print(f"Target  CFG: {G_target.number_of_nodes()} nodes, {G_target.number_of_edges()} edges")

    # Node differences
    missing_in_predict = target_nodes - predict_nodes
    extra_in_predict = predict_nodes - target_nodes

    print(f"\nNodes missing in predict: {sorted(missing_in_predict)}")
    print(f"Extra nodes in predict:    {sorted(extra_in_predict)}")

    # Edge differences
    missing_edges = target_edges - predict_edges
    extra_edges = predict_edges - target_edges

    print(f"\nEdges missing in predict:")
    for (src, dst) in sorted(missing_edges):
        print(f"  {src} -> {dst}")
    print(f"Extra edges in predict:")
    for (src, dst) in sorted(extra_edges):
        print(f"  {src} -> {dst}")

    # Optionally, structural isomorphism ignoring node labels.
    GM = nx.is_isomorphic(G_predict, G_target)
    print(f"\nGraphs isomorphic (ignoring node labels): {GM}")

    # # Optionally, evaluate strongly connected components
    # pred_sccs = list(nx.strongly_connected_components(G_predict))
    # targ_sccs = list(nx.strongly_connected_components(G_target))
    # print(f"Predict CFG has {len(pred_sccs)} strongly connected components.")
    # print(f"Target  CFG has {len(targ_sccs)} strongly connected components.")

    # # Optionally, print the basic blocks for reference
    # pred_blocks = predict_func_data.get("basic_blocks", {})
    # targ_blocks = target_func_data.get("basic_blocks", {})
    # print(f"\nPredict basic blocks (showing first 3):")
    # for k in sorted(pred_blocks)[:3]:
    #     print(f"  {k}: {pred_blocks[k]}")
    # print(f"Target basic blocks (showing first 3):")
    # for k in sorted(targ_blocks)[:3]:
    #     print(f"  {k}: {targ_blocks[k]}")

    # print("\nCFG structure comparison complete.")



def analyze_one_llm_decompile_record(llm_decompile_record: LLMDecompileRecord):
    """Analyze CFG differences between execution success prediction and target."""
    
    # 1. Find the execution success evaluation result and its index
    exec_success_result = None
    retry_count = None
    predict_idx = None
    
    for retry_idx, response_validation in llm_decompile_record.retry_response_validation.items():
        for idx, eval_result in enumerate(response_validation.predict_evaluation_results_list):
            if eval_result.execution_success:
                exec_success_result = eval_result
                retry_count = retry_idx
                predict_idx = idx
                break
        if exec_success_result:
            break
    
    if exec_success_result is None:
        print(f"Record {llm_decompile_record.idx}: No execution success prediction found.")
        return
    
    # 2. Get the sample directory based on retry_count
    if retry_count == -1:
        sample_dir = os.path.join(llm_decompile_record.output_dir, f"sample_{llm_decompile_record.idx}")
    else:
        sample_dir = os.path.join(llm_decompile_record.output_dir, f"sample_{llm_decompile_record.idx}_retry_{retry_count}")
    
    if not os.path.exists(sample_dir):
        print(f"Record {llm_decompile_record.idx}: Sample directory not found: {sample_dir}")
        return
    
    # 3. Get the function name
    func_name = llm_decompile_record.record["func_info"]["functions"][0]["name"]
    
    # 4. Find or create predict.o for the execution success prediction
    compile_dir = os.path.join(sample_dir, str(predict_idx))
    predict_o_path = os.path.join(compile_dir, "predict.o")
    
    if not os.path.exists(predict_o_path):
        # Try to compile it
        from utils.evaluate_exebench import compile_llvm_ir
        success, assembly_path, error_msg = compile_llvm_ir(
            exec_success_result.llvm_ir, compile_dir, name_hint="predict"
        )
        if success:
            predict_o_path = os.path.join(compile_dir, "predict.o")
        else:
            print(f"Record {llm_decompile_record.idx}: Failed to compile predict LLVM IR: {error_msg}")
            return
    
    if not os.path.exists(predict_o_path):
        print(f"Record {llm_decompile_record.idx}: Could not find or create predict.o file at {predict_o_path}")
        return
    
    # 5. Find or create target.o
    target_o_path = os.path.join(sample_dir, "target.o")
    if not os.path.exists(target_o_path):
        # Try to compile target LLVM IR if .o doesn't exist
        from utils.evaluate_exebench import compile_llvm_ir
        target_llvm_ir = llm_decompile_record.record["llvm_ir"]["code"][-1]
        success, assembly_path, error_msg = compile_llvm_ir(
            target_llvm_ir, sample_dir, name_hint="target"
        )
        if success:
            target_o_path = os.path.join(sample_dir, "target.o")
        else:
            print(f"Record {llm_decompile_record.idx}: Failed to compile target LLVM IR: {error_msg}")
            return
    
    if not os.path.exists(target_o_path):
        print(f"Record {llm_decompile_record.idx}: Could not find or create target.o file at {target_o_path}")
        return
    
    # 6. Extract CFG JSON from both object files
    print(f"\nRecord {llm_decompile_record.idx}: Extracting CFG from predict.o and target.o...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        predict_json_path = os.path.join(tmp_dir, "predict_cfg.json")
        target_json_path = os.path.join(tmp_dir, "target_cfg.json")
        
        predict_cfg = _extract_cfg_from_object_file(predict_o_path, func_name, predict_json_path)
        target_cfg = _extract_cfg_from_object_file(target_o_path, func_name, target_json_path)
        
        if predict_cfg is None:
            print(f"Record {llm_decompile_record.idx}: Failed to extract CFG from predict.o")
            return
        
        if target_cfg is None:
            print(f"Record {llm_decompile_record.idx}: Failed to extract CFG from target.o")
            return
        
        # 7. Compare CFG structures
        _compare_cfg_structures(predict_cfg, target_cfg, func_name)


if __name__ == "__main__":
    validation_dir = "/data1/xiachunwei/Projects/validation/Qwen3-32B/sample_loops_Qwen3-32B-n8-assembly-without-comments-ghidra-decompile"
    analyze_cfg_exebench(validation_dir)
