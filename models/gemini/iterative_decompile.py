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
from utils.openai_helper import extract_llvm_code_from_response, format_compile_error_prompt, format_execution_error_prompt
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                n=num_generations,
                stream=True,
                timeout=7200
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield full_response
            
            return full_response
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def evaluate_response(self, response_text: str, record) -> Dict:
        """Evaluate the response and return validation results"""
        try:
            # Create a mock response object for evaluation
            class MockResponse:
                def __init__(self, text):
                    self.choices = [MockChoice(text)]
            
            class MockChoice:
                def __init__(self, text):
                    self.message = MockMessage(text)
            
            class MockMessage:
                def __init__(self, text):
                    self.content = text
            
            mock_response = MockResponse(response_text)
            
            # Extract LLVM code
            predict_list = extract_llvm_code_from_response(mock_response)
            
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
            original_asm = f"```assembly\n{original_asm}\n```"
            
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
                        c_func_def = gr.Textbox(label="C Function Definition", lines=3, max_lines=10)
                        original_asm = gr.Markdown(label="Original Assembly")
                        original_llvm = gr.Markdown(label="Original LLVM IR")
                    
                    with gr.TabItem("Similar Record (RAG)"):
                        similar_info = gr.Markdown(label="Similar Record Info")
                
                gr.Markdown("## Generation")
                prompt = gr.Textbox(label="Generated Prompt", lines=10, max_lines=20)
                num_generations = gr.Slider(minimum=1, maximum=16, value=8, step=1, label="Number of Generations")
                generate_btn = gr.Button("Generate Response", variant="primary")
                
                gr.Markdown("## Generated Response")
                response_output = gr.Textbox(label="Response", lines=15, max_lines=30)
                
                gr.Markdown("## Evaluation Results")
                evaluation_results = gr.JSON(label="Evaluation Results")
                
                gr.Markdown("## Predicted Code")
                with gr.Tabs():
                    with gr.TabItem("Predicted LLVM IR"):
                        predicted_llvm = gr.Markdown(label="Predicted LLVM IR")
                    with gr.TabItem("Predicted Assembly"):
                        predicted_asm = gr.Markdown(label="Predicted Assembly")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Error Fixing")
                error_type = gr.Radio(
                    choices=["compile_error", "execution_error"],
                    value="compile_error",
                    label="Error Type"
                )
                error_msg = gr.Textbox(label="Error Message", lines=3)
                fix_prompt = gr.Textbox(label="Fix Prompt", lines=10, max_lines=20)
                generate_fix_btn = gr.Button("Generate Fix", variant="primary")
                
                gr.Markdown("## Fixed Response")
                fixed_response = gr.Textbox(label="Fixed Response", lines=15, max_lines=30)
                fixed_evaluation = gr.JSON(label="Fixed Evaluation Results")
        
        # Event handlers
        def setup_model_handler(model, host, port, qdrant_host, qdrant_port, embedding_url, icl):
            return tool.setup_model(model, host, port, qdrant_host, qdrant_port, embedding_url, icl)
        
        def load_dataset_handler(path):
            return tool.load_dataset(path)
        
        def load_sample_handler(idx):
            return tool.load_sample(idx)
        
        def generate_response_handler(prompt_text, num_gen):
            # Use Gradio's streaming capabilities
            try:
                response_generator = tool.generate_response_stream(prompt_text, num_gen)
                full_response = ""
                for chunk in response_generator:
                    full_response = chunk
                    yield full_response
            except Exception as e:
                yield f"Error generating response: {str(e)}"
        
        def evaluate_response_handler(response_text):
            if tool.current_record:
                return tool.evaluate_response(response_text, tool.current_record)
            return {"error": "No current record loaded"}
        
        def prepare_fix_prompt_handler(error_type, error_msg, predicted_llvm, predicted_asm):
            return tool.prepare_fix_prompt(error_type, error_msg, predicted_llvm, predicted_asm)
        
        def generate_fix_handler(fix_prompt_text, num_gen):
            # Use Gradio's streaming capabilities
            try:
                response_generator = tool.generate_response_stream(fix_prompt_text, num_gen)
                full_response = ""
                for chunk in response_generator:
                    full_response = chunk
                    yield full_response
            except Exception as e:
                yield f"Error generating fix: {str(e)}"
        
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
            outputs=response_output,
            api_name="generate_response"
        )
        
        # Auto-evaluate when response is generated (only after completion)
        def on_response_complete(response_text):
            # Only evaluate if we have a complete response (not streaming)
            if response_text and len(response_text) > 100 and not response_text.startswith("Error"):
                return evaluate_response_handler(response_text)
            return {"status": "Waiting for complete response..."}
        
        response_output.change(
            on_response_complete,
            inputs=response_output,
            outputs=evaluation_results
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
        
        # Prepare fix prompt
        error_type.change(
            prepare_fix_prompt_handler,
            inputs=[error_type, error_msg, predicted_llvm, predicted_asm],
            outputs=fix_prompt
        )
        error_msg.change(
            prepare_fix_prompt_handler,
            inputs=[error_type, error_msg, predicted_llvm, predicted_asm],
            outputs=fix_prompt
        )
        predicted_llvm.change(
            prepare_fix_prompt_handler,
            inputs=[error_type, error_msg, predicted_llvm, predicted_asm],
            outputs=fix_prompt
        )
        predicted_asm.change(
            prepare_fix_prompt_handler,
            inputs=[error_type, error_msg, predicted_llvm, predicted_asm],
            outputs=fix_prompt
        )
        
        generate_fix_btn.click(
            generate_fix_handler,
            inputs=[fix_prompt, num_generations],
            outputs=[fixed_response, fixed_evaluation],
            api_name="generate_fix"
        )
        
        # Fixed response evaluation is now handled by the button click events
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
