#!/usr/bin/env python3
"""
Launcher script for the Interactive Decompilation Tool
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'openai', 
        'datasets',
        'qdrant-client',
        'torch',
        'transformers',
        'sentence-transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment variables and setup"""
    print("Checking environment...")
    
    # Check if we're in the right directory
    current_dir = Path(__file__).parent
    if not (current_dir / "iterative_decompile.py").exists():
        print("Error: iterative_decompile.py not found in current directory")
        return False
    
    # Check environment variables for cloud models
    cloud_models = {
        "ARK_STREAM_API_KEY": "Huoshan-DeepSeek-R1",
        "OPENAI_API_KEY": "OpenAI-GPT-4.1"
    }
    
    for env_var, model in cloud_models.items():
        if not os.environ.get(env_var):
            print(f"Warning: {env_var} not set (required for {model})")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Launch Interactive Decompilation Tool")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Interactive Decompilation Tool Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("✅ Environment check passed")
    print(f"🌐 Starting server on {args.host}:{args.port}")
    if args.share:
        print("🔗 Public link will be created")
    
    # Set debug environment variable if requested
    if args.debug:
        os.environ["GRADIO_DEBUG"] = "1"
        print("🐛 Debug mode enabled")
    
    try:
        # Import and run the tool
        from iterative_decompile import create_gradio_interface
        
        interface = create_gradio_interface()
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Tool stopped by user")
    except Exception as e:
        print(f"❌ Error launching tool: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


