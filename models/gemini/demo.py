#!/usr/bin/env python3
"""
Demo script showing how to use the IterativeDecompilationTool programmatically
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.gemini.iterative_decompile import IterativeDecompilationTool

def demo_basic_usage():
    """Demonstrate basic usage of the tool"""
    print("🔧 Interactive Decompilation Tool Demo")
    print("=" * 50)
    
    # Initialize the tool
    tool = IterativeDecompilationTool()
    
    # Setup model (using local Qwen3-32B as example)
    print("1. Setting up model...")
    status = tool.setup_model(
        model_name="Qwen3-32B",
        host="localhost",
        port="9001",
        in_context_learning=True
    )
    print(f"   Status: {status}")
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset_path = f"{os.path.expanduser('~')}/Datasets/filtered_exebench/sampled_dataset_with_loops_164"
    if os.path.exists(dataset_path):
        status = tool.load_dataset(dataset_path)
        print(f"   Status: {status}")
    else:
        print(f"   Dataset not found at: {dataset_path}")
        print("   Please update the dataset path in the script")
        return
    
    # Load a sample
    print("\n3. Loading sample...")
    c_func_def, original_asm, original_llvm, prompt, similar_info = tool.load_sample(0)
    print(f"   C Function: {c_func_def[:100]}...")
    print(f"   Original Assembly: {len(original_asm)} characters")
    print(f"   Original LLVM IR: {len(original_llvm)} characters")
    print(f"   Prompt: {len(prompt)} characters")
    
    # Generate response (if model is available)
    print("\n4. Generating response...")
    try:
        response_generator = tool.generate_response_stream(prompt, num_generations=1)
        response = ""
        for chunk in response_generator:
            response = chunk
            print(f"   Generated: {len(response)} characters...")
        
        print(f"   Final response: {len(response)} characters")
        
        # Evaluate response
        print("\n5. Evaluating response...")
        evaluation = tool.evaluate_response(response, tool.current_record)
        print(f"   Evaluation: {evaluation}")
        
    except Exception as e:
        print(f"   Error generating response: {e}")
        print("   This is expected if the model server is not running")
    
    print("\n✅ Demo completed!")
    print("\nTo run the full interactive tool:")
    print("   python launch_tool.py")

def demo_error_fixing():
    """Demonstrate error fixing functionality"""
    print("\n🔧 Error Fixing Demo")
    print("=" * 30)
    
    tool = IterativeDecompilationTool()
    
    # Load a sample
    dataset_path = f"{os.path.expanduser('~')}/Datasets/filtered_exebench/sampled_dataset_with_loops_164"
    if os.path.exists(dataset_path):
        tool.load_dataset(dataset_path)
        tool.load_sample(0)
        
        # Demonstrate fix prompt generation
        print("1. Generate compile error fix prompt...")
        compile_fix_prompt = tool.prepare_fix_prompt(
            error_type="compile_error",
            error_msg="error: expected instruction opcode",
            predicted_llvm="define i32 @func() { ret i32 0 }"
        )
        print(f"   Compile fix prompt: {len(compile_fix_prompt)} characters")
        
        print("\n2. Generate execution error fix prompt...")
        exec_fix_prompt = tool.prepare_fix_prompt(
            error_type="execution_error",
            predicted_llvm="define i32 @func() { ret i32 0 }",
            predicted_asm="mov eax, 0\nret"
        )
        print(f"   Execution fix prompt: {len(exec_fix_prompt)} characters")
    
    print("\n✅ Error fixing demo completed!")

if __name__ == "__main__":
    demo_basic_usage()
    demo_error_fixing()
