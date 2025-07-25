import os
import pickle
import subprocess
from datasets import load_from_disk
import logging
from flask import Flask, render_template, jsonify, request
from models.gemini.gemini_decompilation import extract_llvm_code_from_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store the loaded data
model = "qwen3-32b"
dataset_paires = [
    ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_164", f"validation/{model}/sample_loops_{model}-n8-assembly-with-comments"),
    ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164", f"validation/{model}/sample_without_loops_{model}-n8-assembly-with-comments"),
    ("/home/xiachunwei/Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164", f"validation/{model}/sample_only_one_bb_{model}-n8-assembly-with-comments"),
    ("/home/xiachunwei/Datasets/filtered_exebench/sampled_on_bb_function_call", f"validation/{model}/sample_one_bb_with_functions_{model}-n8-assembly-with-comments"),
    ("/home/xiachunwei/Datasets/filtered_exebench/sampled_on_bb_without_function_call", f"validation/{model}/sample_one_bb_wo_functions_{model}-n8-assembly-with-comments")
]

def compile_llvm_ir(llvm_ir: str, compile_dir: str, name_hint)->tuple[str]:
    """Compile the llvm_ir to assembly and save the results to the validation directory, return true if success compile
    Args:
        llvm_ir: str, the llvm ir code.
        compile_dir: str, the directory to save the compiled assembly code.
        name_hint: str, the hint for the name of the compiled file.
    
    Returns:
        assembly or error_mes: str
    """

    llvm_ir_path = os.path.join(compile_dir, f"{name_hint}.ll")
    assembly_path = os.path.join(compile_dir, f"{name_hint}.s")
    if isinstance(llvm_ir, list) and len(llvm_ir) == 0:
        return None, None
    with open(llvm_ir_path, 'w') as f:
        f.write(llvm_ir[0] if isinstance(llvm_ir, list) else llvm_ir)
    try:
        # 3. Compile the llvm ir to assembly
        cmd = ["llc", llvm_ir_path, "-o", assembly_path]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode == 0:
            with open(assembly_path, 'r') as f:
                return f.read()
        else:
            return ret.stderr.decode()
    except Exception as e:
        logger.warning("Error compiling")
    return ""


# Load the first dataset by default
current_dataset_idx = 2
results, exebench_dataset = None, None
current_predict_idx = 0


def load_content(dataset_dir_path, response_output_dir):
    results = pickle.load(open(f"{response_output_dir}/results.pkl", "rb"))
    exebench_dataset = load_from_disk(dataset_dir_path)
    return results, exebench_dataset

def get_idx_from_results(results, exebench_dataset, idx):
    response, eval_result = results[idx]
    record = exebench_dataset[idx]
    original_llvm_ir = record["llvm_ir"]['code'][-1]
    original_asm_code = record["asm"]["code"][-1]
    predict_list = extract_llvm_code_from_response(response)
    
    predicted_llvm_ir = predict_list[current_predict_idx] if predict_list else "No LLVM IR prediction available"
    predicted_llvm_ir = predicted_llvm_ir[0] if predicted_llvm_ir is not None else "No LLVM IR prediction available"
    predicted_asm_code = compile_llvm_ir(predicted_llvm_ir, "/tmp/", "predict")
    predict_compile_success = eval_result["predict_compile_success"]
    predict_execution_success = eval_result["predict_execution_success"]
    target_compile_success = eval_result["target_compile_success"]
    target_execution_success = eval_result["target_execution_success"]
    return original_llvm_ir, original_asm_code, predicted_llvm_ir, predicted_asm_code, predict_compile_success, predict_execution_success, target_compile_success, target_execution_success

@app.route('/')
def index():
    global results, exebench_dataset
    if results is None or exebench_dataset is None:
        dataset_dir_path, response_output_dir = dataset_paires[current_dataset_idx]
        results, exebench_dataset = load_content(dataset_dir_path, response_output_dir)
    return render_template('index.html')

@app.route('/update_predict_idx', methods=['POST'])
def update_predict_idx():
    global current_predict_idx
    try:
        idx = int(request.json.get('idx', 0))
        current_predict_idx = idx
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_by_index', methods=['POST'])
def load_by_index():
    global results, exebench_dataset
    try:
        idx = int(request.json.get('idx', 0))
        if results is None or exebench_dataset is None:
            dataset_dir_path, response_output_dir = dataset_paires[current_dataset_idx]
            results, exebench_dataset = load_content(dataset_dir_path, response_output_dir)
        
        original_llvm_ir, original_asm_code, predicted_llvm_ir, predicted_asm_code, predict_compile_success, predict_execution_success, target_compile_success, target_execution_success = get_idx_from_results(results, exebench_dataset, idx)
        
        # Extract all predicted IR and assembly code
        response, _ = results[idx]
        predict_list = extract_llvm_code_from_response(response)
        predicted_ir_list = predict_list if predict_list else []
        predicted_asm_list = []
        
        # Get raw message content
        if response.choices and len(response.choices) > current_predict_idx and response.choices[current_predict_idx].message.model_extra:
            if 'reasoning_content' in response.choices[current_predict_idx].message.model_extra.keys():
                raw_message = response.choices[current_predict_idx].message.model_extra['reasoning_content']
        else:
            raw_message = "reasoning content is not avaliable"
        
        # Generate assembly code for each predicted IR
        for ir in predicted_ir_list:
            asm = compile_llvm_ir(ir, "/tmp/", f"predict_{len(predicted_asm_list)}")
            predicted_asm_list.append(asm if asm else "")
        
        return jsonify({
            'success': True,
            'original_asm': original_asm_code,
            'predicted_asm': predicted_asm_code,
            'original_ir': original_llvm_ir,
            'predicted_ir': predicted_llvm_ir,
            'predicted_ir_list': predicted_ir_list,
            'predicted_asm_list': predicted_asm_list,
            'predict_compile_success': predict_compile_success,
            'predict_execution_success': predict_execution_success,
            'target_compile_success': target_compile_success,
            'target_execution_success': target_execution_success,
            'current_predict_idx': current_predict_idx,
            'raw_message': raw_message
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_original_asm')
def load_original_asm():
    try:
        with open('original.asm', 'r') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_predicted_asm')
def load_predicted_asm():
    try:
        with open('predicted.asm', 'r') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_original_ir')
def load_original_ir():
    try:
        with open('original.ll', 'r') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_predicted_ir')
def load_predicted_ir():
    try:
        with open('predicted.ll', 'r') as f:
            content = f.read()
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
