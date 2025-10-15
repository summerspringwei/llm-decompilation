import os
import pickle
import logging
import tempfile
import gradio as gr
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from datasets import load_from_disk, Dataset
from qdrant_client import QdrantClient

from utils.evaluate_exebench import compile_llvm_ir, eval_assembly
from utils.preprocessing_assembly import preprocessing_assembly
from utils.preprocessing_llvm_ir import preprocessing_llvm_ir
from utils.openai_helper import extrac_llvm_code_from_response_text, format_compile_error_prompt, format_execution_error_prompt
from models.rag.exebench_qdrant_base import load_embedding_model, find_similar_records_in_exebench_synth_rich_io
from utils.openai_helper import GENERAL_INIT_PROMPT, SIMILAR_RECORD_PROMPT
from models.rag.embedding_client import RemoteEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s: %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = os.path.expanduser("~")

class IterativeDecompilationTool:
    def __init__(self):
        self.dataset = None
        self.current_idx = 0
        self.current_record = None
        self.current_response = None
        self.current_validation = None
        self.similar_record = None
        self.embedding_model = None
        self.qdrant_client = None
        self.client = None
        self.model_name = None
        self.in_context_learning = True
        self.remove_comments = True
        self.temp_dir = tempfile.mkdtemp()
        
        # Service configuration
        self.SERVICE_CONFIG = {
            "Qwen3-32B": (OpenAI(
                api_key="token-llm4decompilation-abc123",
                base_url="http://localhost:9001/v1",
            ), "Qwen3-32B", GENERAL_INIT_PROMPT),
            "Qwen3-30B-A3B": (OpenAI(
                api_key="token-llm4decompilation-abc123",
                base_url="http://localhost:9001/v1",
            ), "Qwen3-30B-A3B", GENERAL_INIT_PROMPT),
            "Huoshan-DeepSeek-R1": (OpenAI(
                api_key=os.environ.get("ARK_STREAM_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                timeout=1800
            ), "ep-20250317013717-m9ksl", GENERAL_INIT_PROMPT),
            "OpenAI-GPT-4.1": (OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            ), "gpt-4.1", GENERAL_INIT_PROMPT)
        }
        
        # Qdrant configuration
        self.dataset_for_qdrant_dir = f"{HOME_DIR}/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"

    def setup_model(self, model_name: str, host: str = "localhost", port: str = "9001", 
                   qdrant_host: str = "localhost", qdrant_port: str = "6333",
                   embedding_url: str = "http://localhost:8001",
                   in_context_learning: bool = True):
        """Setup the model and connections"""
        try:
            # Update service config with custom host/port
            if model_name in ["Qwen3-32B", "Qwen3-30B-A3B"]:
                self.SERVICE_CONFIG[model_name] = (OpenAI(
                    api_key="token-llm4decompilation-abc123",
                    base_url=f"http://{host}:{port}/v1",
                ), model_name, GENERAL_INIT_PROMPT)
            
            self.client, self.model_name, _ = self.SERVICE_CONFIG[model_name]
            self.in_context_learning = in_context_learning
            
            # Setup Qdrant
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            # Setup embedding model for RAG
            if in_context_learning:
                self.embedding_model = RemoteEmbeddingModel(embedding_url)
            
            return f"Model {model_name} setup successfully!"
        except Exception as e:
            return f"Error setting up model: {str(e)}"

    def load_dataset(self, dataset_path: str) -> str:
        """Load dataset from path"""
        try:
            self.dataset = load_from_disk(dataset_path)
            return f"Dataset loaded successfully! Total samples: {len(self.dataset)}"
        except Exception as e:
            return f"Error loading dataset: {str(e)}"

    def prepare_prompt(self, record, remove_comments: bool = True):
        """Prepare prompt for decompilation"""
        if self.in_context_learning and self.embedding_model:
            return self.prepare_prompt_from_similar_record(record, remove_comments)
        else:
            return self.prepare_prompt_basic(record, remove_comments)

    def prepare_prompt_basic(self, record, remove_comments: bool = True):
        """Prepare basic prompt without similar records"""
        asm_code = record["asm"]["code"][-1]
        asm_code = preprocessing_assembly(asm_code, remove_comments=remove_comments)
        return GENERAL_INIT_PROMPT.format(asm_code=asm_code)

    def prepare_prompt_from_similar_record(self, record, remove_comments: bool = True):
        """Prepare prompt with similar record for in-context learning"""
        similar_record, score = find_similar_records_in_exebench_synth_rich_io(
            self.qdrant_client, self.embedding_model, record, self.dataset_for_qdrant_dir
        )
        self.similar_record = similar_record
        
        asm_code = record["asm"]["code"][-1]
        asm_code = preprocessing_assembly(asm_code, remove_comments=remove_comments)
        similar_asm_code = similar_record["asm"]["code"][-1]
        similar_asm_code = preprocessing_assembly(similar_asm_code, remove_comments=remove_comments)
        similar_llvm_ir = similar_record["llvm_ir"]["code"][-1]
        similar_llvm_ir = preprocessing_llvm_ir(similar_llvm_ir)
        
        return SIMILAR_RECORD_PROMPT.format(
            asm_code=asm_code, 
            similar_asm_code=similar_asm_code, 
            similar_llvm_ir=similar_llvm_ir
        )

    def generate_response_stream(self, prompt: str, num_generations: int = 8):
        """Generate response with streaming"""
        try:
            # When n > 1, streaming typically doesn't work well
            # Generate all completions without streaming
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                n=num_generations,
                stream=False,
                timeout=7200
            )
            
            # Extract all completions
            responses = []
            for choice in response.choices:
                responses.append(choice.message.content)
            
            return responses
        except Exception as e:
            return [f"Error generating response: {str(e)}"]

    def evaluate_response(self, response_text_list: List[str], record) -> Dict:
        """Evaluate the response and return validation results"""
        try:
            # Extract LLVM code
            predict_list = [extrac_llvm_code_from_response_text(response_text) for response_text in response_text_list]
            
            # Create sample directory
            sample_dir = os.path.join(self.temp_dir, f"sample_{self.current_idx}")
            os.makedirs(sample_dir, exist_ok=True)
            
            predict_compile_success_list = []
            predict_execution_success_list = []
            error_msg_list = []
            
            # Evaluate each prediction
            for predict in predict_list:
                predict_execution_success = False
                predict_compile_success, predict_assembly_path, error_msg = compile_llvm_ir(
                    predict, sample_dir, name_hint="predict"
                )
                if predict_compile_success:
                    with open(predict_assembly_path, 'r') as f:
                        predict_execution_success = eval_assembly(record, f.read())
                
                predict_compile_success_list.append(predict_compile_success)
                predict_execution_success_list.append(predict_execution_success)
                error_msg_list.append(error_msg)
            
            # Compile target LLVM IR
            target_compile_success, target_assembly_path, target_error_msg = compile_llvm_ir(
                record["llvm_ir"]["code"][-1], sample_dir, name_hint="target"
            )
            target_execution_success = False
            if target_compile_success:
                with open(target_assembly_path, 'r') as f:
                    target_execution_success = eval_assembly(record, f.read())
            
            return {
                "predict_compile_success": predict_compile_success_list,
                "predict_execution_success": predict_execution_success_list,
                "predict_error_msg": error_msg_list,
                "target_compile_success": target_compile_success,
                "target_execution_success": target_execution_success,
                "predict_list": predict_list
            }
        except Exception as e:
            return {"error": str(e)}

    def load_sample(self, idx: int) -> Tuple[str, str, str, str, str]:
        """Load a sample by index"""
        if self.dataset is None:
            return "Please load a dataset first", "", "", "", ""
        
        try:
            idx = int(idx)
            if idx >= len(self.dataset):
                return f"Index {idx} out of range. Dataset has {len(self.dataset)} samples", "", "", "", ""
            
            self.current_idx = idx
            self.current_record = self.dataset[idx]
            
            # Get original C function definition
            c_func_def = self.current_record.get("func_def", "No C function definition available")
            
            # Get original assembly
            original_asm = self.current_record["asm"]["code"][-1]
            original_asm = preprocessing_assembly(original_asm, remove_comments=self.remove_comments)
            original_asm = f"{original_asm}"
            
            # Get original LLVM IR
            original_llvm = self.current_record["llvm_ir"]["code"][-1]
            original_llvm = preprocessing_llvm_ir(original_llvm)
            original_llvm = f"```llvm\n{original_llvm}\n```"
            
            # Prepare prompt
            prompt = self.prepare_prompt(self.current_record, self.remove_comments)
            
            # Get similar record info if available
            similar_info = ""
            if self.similar_record:
                similar_asm = self.similar_record["asm"]["code"][-1]
                similar_asm = preprocessing_assembly(similar_asm, remove_comments=self.remove_comments)
                similar_llvm = self.similar_record["llvm_ir"]["code"][-1]
                similar_llvm = preprocessing_llvm_ir(similar_llvm)
                similar_info = f"**Similar Assembly:**\n```assembly\n{similar_asm}\n```\n\n**Similar LLVM IR:**\n```llvm\n{similar_llvm}\n```"
            
            return c_func_def, original_asm, original_llvm, prompt, similar_info
            
        except Exception as e:
            return f"Error loading sample: {str(e)}", "", "", "", ""

    def prepare_fix_prompt(self, error_type: str, error_msg: str = "", predicted_llvm: str = "", 
                          predicted_asm: str = "") -> str:
        """Prepare prompt for fixing errors"""
        if not self.current_record:
            return "No current record loaded"
        
        target_asm_code = self.current_record["asm"]["code"][-1]
        target_asm_code = preprocessing_assembly(target_asm_code, remove_comments=self.remove_comments)
        
        if error_type == "compile_error":
            return format_compile_error_prompt(
                target_asm_code, predicted_llvm, error_msg, 
                self.in_context_learning, 
                self.similar_record["asm"]["code"][-1] if self.similar_record else None,
                self.similar_record["llvm_ir"]["code"][-1] if self.similar_record else None
            )
        elif error_type == "execution_error":
            return format_execution_error_prompt(
                target_asm_code, predicted_llvm, predicted_asm,
                self.in_context_learning,
                self.similar_record["asm"]["code"][-1] if self.similar_record else None,
                self.similar_record["llvm_ir"]["code"][-1] if self.similar_record else None
            )
        else:
            return "Invalid error type"

def create_gradio_interface():
    """Create the Gradio interface"""
    tool = IterativeDecompilationTool()
    
    with gr.Blocks(title="Iterative Decompilation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔧 Interactive Decompilation Tool")
        gr.Markdown("Load datasets, generate LLVM IR from assembly, and iteratively fix errors")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Model Configuration")
                
                model_name = gr.Dropdown(
                    choices=["Qwen3-32B", "Qwen3-30B-A3B", "Huoshan-DeepSeek-R1", "OpenAI-GPT-4.1"],
                    value="Qwen3-32B",
                    label="Model"
                )
                
                host = gr.Textbox(value="localhost", label="Host")
                port = gr.Textbox(value="9001", label="Port")
                qdrant_host = gr.Textbox(value="localhost", label="Qdrant Host")
                qdrant_port = gr.Textbox(value="6333", label="Qdrant Port")
                embedding_url = gr.Textbox(
                    value="http://localhost:8001",
                    label="Embedding Model URL"
                )
                in_context_learning = gr.Checkbox(value=True, label="Use In-Context Learning")
                
                setup_btn = gr.Button("Setup Model", variant="primary")
                setup_status = gr.Textbox(label="Setup Status", interactive=False)
                
                gr.Markdown("## Dataset")
                dataset_path = gr.Textbox(
                    value=f"{HOME_DIR}/Datasets/filtered_exebench/sampled_dataset_with_loops_164",
                    label="Dataset Path"
                )
                load_dataset_btn = gr.Button("Load Dataset", variant="primary")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False)
                
                gr.Markdown("## Sample Selection")
                sample_idx = gr.Number(value=0, label="Sample Index", minimum=0)
                load_sample_btn = gr.Button("Load Sample", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("## Current Sample")
                
                with gr.Tabs():
                    with gr.TabItem("Original Code"):
                        c_func_def = gr.Code(label="C Function Definition", lines=3, max_lines=100, language="c")
                        original_asm = gr.Code(label="Original Assembly", lines=3, max_lines=100, language="markdown")
                        original_llvm = gr.Code(label="Original LLVM IR", lines=3, max_lines=100, language="markdown")
                    
                    with gr.TabItem("Similar Record (RAG)"):
                        similar_info = gr.Markdown(label="Similar Record Info")
                
                gr.Markdown("## Generation")
                prompt = gr.Textbox(label="Generated Prompt", lines=10, max_lines=20)
                num_generations = gr.Slider(minimum=1, maximum=8, value=8, step=1, label="Number of Generations")
                generate_btn = gr.Button("Generate Response", variant="primary")
                
                gr.Markdown("## Generated Responses")
                with gr.Tabs():
                    response_tab_1 = gr.TabItem("Generation 1")
                    with response_tab_1:
                        response_1 = gr.Markdown(label="Response 1")
                    
                    response_tab_2 = gr.TabItem("Generation 2")
                    with response_tab_2:
                        response_2 = gr.Markdown(label="Response 2")
                    
                    response_tab_3 = gr.TabItem("Generation 3")
                    with response_tab_3:
                        response_3 = gr.Markdown(label="Response 3")
                    
                    response_tab_4 = gr.TabItem("Generation 4")
                    with response_tab_4:
                        response_4 = gr.Markdown(label="Response 4")
                    
                    response_tab_5 = gr.TabItem("Generation 5")
                    with response_tab_5:
                        response_5 = gr.Markdown(label="Response 5")
                    
                    response_tab_6 = gr.TabItem("Generation 6")
                    with response_tab_6:
                        response_6 = gr.Markdown(label="Response 6")
                    
                    response_tab_7 = gr.TabItem("Generation 7")
                    with response_tab_7:
                        response_7 = gr.Markdown(label="Response 7")
                    
                    response_tab_8 = gr.TabItem("Generation 8")
                    with response_tab_8:
                        response_8 = gr.Markdown(label="Response 8")
                
                gr.Markdown("## Evaluation")
                evaluate_btn = gr.Button("Evaluate Responses", variant="primary")
                evaluation_results = gr.JSON(label="Evaluation Results")
                
                gr.Markdown("## Predicted Code")
                with gr.Tabs():
                    with gr.TabItem("Predicted LLVM IR"):
                        predicted_llvm = gr.Markdown(label="Predicted LLVM IR")
                    with gr.TabItem("Predicted Assembly"):
                        predicted_asm = gr.Markdown(label="Predicted Assembly")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Evaluation Results Summary")
                eval_summary = gr.Markdown(label="Evaluation Summary", value="No evaluation results yet")
                
                gr.Markdown("## Error Fixing")
                selected_generation = gr.Dropdown(
                    choices=[f"Generation {i+1}" for i in range(8)],
                    value="Generation 1",
                    label="Select Generation to Fix"
                )
                prepare_fix_btn = gr.Button("Prepare Fix Prompt", variant="secondary")
                fix_prompt = gr.Textbox(label="Fix Prompt", lines=10, max_lines=20, interactive=True)
                generate_fix_btn = gr.Button("Generate Fix", variant="primary")
        
        # Event handlers
        def setup_model_handler(model, host, port, qdrant_host, qdrant_port, embedding_url, icl):
            return tool.setup_model(model, host, port, qdrant_host, qdrant_port, embedding_url, icl)
        
        def load_dataset_handler(path):
            return tool.load_dataset(path)
        
        def load_sample_handler(idx):
            return tool.load_sample(idx)
        
        def generate_response_handler(prompt_text, num_gen):
            # Generate all responses
            try:
                responses = tool.generate_response_stream(prompt_text, int(num_gen))
                
                # Format each response as markdown code block
                formatted_responses = []
                for i, response in enumerate(responses):
                    formatted_responses.append(f"```llvm\n{response}\n```")
                
                # Pad with empty strings if less than 8 responses
                while len(formatted_responses) < 8:
                    formatted_responses.append("")
                
                # Return exactly 8 outputs for the 8 tabs
                return formatted_responses[0], formatted_responses[1], formatted_responses[2], formatted_responses[3], formatted_responses[4], formatted_responses[5], formatted_responses[6], formatted_responses[7]
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                return error_msg, "", "", "", "", "", "", ""
        
        def evaluate_response_handler(resp1, resp2, resp3, resp4, resp5, resp6, resp7, resp8):
            if not tool.current_record:
                return {"error": "No current record loaded"}, "❌ No current record loaded"
            
            # Collect non-empty responses
            all_responses = [resp1, resp2, resp3, resp4, resp5, resp6, resp7, resp8]
            non_empty_responses = [r for r in all_responses if r and r.strip() != ""]
            
            if len(non_empty_responses) == 0:
                return {"error": "No responses to evaluate"}, "❌ No responses to evaluate"
            
            eval_results = tool.evaluate_response(non_empty_responses, tool.current_record)
            
            # Generate summary with icons and colors
            if "error" in eval_results:
                summary = f"❌ **Error:** {eval_results['error']}"
            else:
                summary = "## Evaluation Results\n\n"
                summary += "| Generation | Compile | Execute | Error Message |\n"
                summary += "|------------|---------|---------|---------------|\n"
                
                compile_success_list = eval_results.get("predict_compile_success", [])
                execution_success_list = eval_results.get("predict_execution_success", [])
                error_msg_list = eval_results.get("predict_error_msg", [])
                
                for i in range(len(compile_success_list)):
                    gen_num = i + 1
                    compile_icon = "✅" if compile_success_list[i] else "❌"
                    execute_icon = "✅" if execution_success_list[i] else "❌"
                    error_msg = error_msg_list[i] if i < len(error_msg_list) else ""
                    error_preview = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                    summary += f"| Generation {gen_num} | {compile_icon} | {execute_icon} | {error_preview} |\n"
                
                # Add target summary
                target_compile = "✅" if eval_results.get("target_compile_success", False) else "❌"
                target_execute = "✅" if eval_results.get("target_execution_success", False) else "❌"
                summary += f"\n**Target LLVM:** Compile {target_compile} | Execute {target_execute}\n"
            
            return eval_results, summary
        
        def prepare_fix_prompt_handler(selected_gen, eval_results, resp1, resp2, resp3, resp4, resp5, resp6, resp7, resp8):
            """Prepare fix prompt based on selected generation"""
            if not tool.current_record:
                return "❌ No current record loaded"
            
            if not eval_results or "error" in eval_results:
                return "❌ Please evaluate responses first"
            
            # Parse selected generation number
            gen_num = int(selected_gen.split()[-1]) - 1  # "Generation 1" -> 0
            
            # Get the selected response
            all_responses = [resp1, resp2, resp3, resp4, resp5, resp6, resp7, resp8]
            selected_response = all_responses[gen_num]
            
            if not selected_response or selected_response.strip() == "":
                return "❌ Selected generation is empty"
            
            # Get evaluation results for this generation
            compile_success_list = eval_results.get("predict_compile_success", [])
            execution_success_list = eval_results.get("predict_execution_success", [])
            error_msg_list = eval_results.get("predict_error_msg", [])
            predict_list = eval_results.get("predict_list", [])
            
            if gen_num >= len(compile_success_list):
                return "❌ No evaluation results for this generation"
            
            compile_success = compile_success_list[gen_num]
            execution_success = execution_success_list[gen_num]
            error_msg = error_msg_list[gen_num] if gen_num < len(error_msg_list) else ""
            predicted_llvm = predict_list[gen_num] if gen_num < len(predict_list) else ""
            
            # Determine error type and prepare prompt
            target_asm_code = tool.current_record["asm"]["code"][-1]
            target_asm_code = preprocessing_assembly(target_asm_code, remove_comments=tool.remove_comments)
            
            if not compile_success:
                # Compile error
                return format_compile_error_prompt(
                    target_asm_code, predicted_llvm, error_msg,
                    tool.in_context_learning,
                    tool.similar_record["asm"]["code"][-1] if tool.similar_record else None,
                    tool.similar_record["llvm_ir"]["code"][-1] if tool.similar_record else None
                )
            elif not execution_success:
                # Execution error - need to get predicted assembly
                try:
                    sample_dir = os.path.join(tool.temp_dir, f"sample_{tool.current_idx}")
                    success, asm_path, _ = compile_llvm_ir(predicted_llvm, sample_dir, f"predict_gen_{gen_num}")
                    if success and asm_path:
                        with open(asm_path, 'r') as f:
                            predicted_asm = f.read()
                    else:
                        predicted_asm = ""
                except:
                    predicted_asm = ""
                
                return format_execution_error_prompt(
                    target_asm_code, predicted_llvm, predicted_asm,
                    tool.in_context_learning,
                    tool.similar_record["asm"]["code"][-1] if tool.similar_record else None,
                    tool.similar_record["llvm_ir"]["code"][-1] if tool.similar_record else None
                )
            else:
                return "✅ This generation compiled and executed successfully! No fix needed."
        
        def generate_fix_handler(fix_prompt_text, num_gen):
            # Generate fix responses
            try:
                responses = tool.generate_response_stream(fix_prompt_text, int(num_gen))
                
                # Format responses as markdown
                formatted_output = ""
                for i, response in enumerate(responses, 1):
                    formatted_output += f"### Fix Generation {i}\n\n"
                    formatted_output += f"```\n{response}\n```\n\n"
                    formatted_output += "---\n\n"
                
                # Evaluate the fix
                combined_response = "\n\n".join(responses)
                if tool.current_record:
                    eval_results = tool.evaluate_response(combined_response, tool.current_record)
                else:
                    eval_results = {"error": "No current record loaded"}
                
                return formatted_output, eval_results
            except Exception as e:
                error_msg = f"Error generating fix: {str(e)}"
                return error_msg, {"error": str(e)}
        
        # Connect events
        setup_btn.click(
            setup_model_handler,
            inputs=[model_name, host, port, qdrant_host, qdrant_port, embedding_url, in_context_learning],
            outputs=setup_status
        )
        
        load_dataset_btn.click(
            load_dataset_handler,
            inputs=dataset_path,
            outputs=dataset_status
        )
        
        load_sample_btn.click(
            load_sample_handler,
            inputs=sample_idx,
            outputs=[c_func_def, original_asm, original_llvm, prompt, similar_info]
        )
        
        generate_btn.click(
            generate_response_handler,
            inputs=[prompt, num_generations],
            outputs=[response_1, response_2, response_3, response_4, response_5, response_6, response_7, response_8],
            api_name="generate_response"
        )
        
        # Evaluate button click
        evaluate_btn.click(
            evaluate_response_handler,
            inputs=[response_1, response_2, response_3, response_4, response_5, response_6, response_7, response_8],
            outputs=[evaluation_results, eval_summary]
        )
        
        # Prepare fix prompt button
        prepare_fix_btn.click(
            prepare_fix_prompt_handler,
            inputs=[selected_generation, evaluation_results, response_1, response_2, response_3, response_4, response_5, response_6, response_7, response_8],
            outputs=fix_prompt
        )
        
        # Update predicted code when evaluation changes
        def update_predicted_code(eval_results):
            if isinstance(eval_results, dict) and "predict_list" in eval_results:
                predict_list = eval_results["predict_list"]
                if predict_list and len(predict_list) > 0:
                    # Get the first successful prediction or the first one
                    predicted_llvm_text = predict_list[0] if isinstance(predict_list[0], str) else str(predict_list[0])
                    
                    # Format LLVM IR as markdown
                    predicted_llvm_markdown = f"```llvm\n{predicted_llvm_text}\n```"
                    
                    # Try to compile and get assembly
                    try:
                        sample_dir = os.path.join(tool.temp_dir, f"sample_{tool.current_idx}")
                        os.makedirs(sample_dir, exist_ok=True)
                        success, asm_path, _ = compile_llvm_ir(predicted_llvm_text, sample_dir, "predicted")
                        if success and asm_path:
                            with open(asm_path, 'r') as f:
                                predicted_asm_text = f.read()
                            predicted_asm_markdown = f"```assembly\n{predicted_asm_text}\n```"
                        else:
                            predicted_asm_markdown = "**Compilation failed**"
                    except:
                        predicted_asm_markdown = "**Error generating assembly**"
                    
                    return predicted_llvm_markdown, predicted_asm_markdown
            return "", ""
        
        evaluation_results.change(
            update_predicted_code,
            inputs=evaluation_results,
            outputs=[predicted_llvm, predicted_asm]
        )
        
        # Generate fix button
        generate_fix_btn.click(
            generate_response_handler,
            inputs=[fix_prompt, num_generations],
            outputs=[response_1, response_2, response_3, response_4, response_5, response_6, response_7, response_8],
            api_name="generate_fix"
        )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
