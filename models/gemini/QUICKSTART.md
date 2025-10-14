# Quick Start Guide

Get the Interactive Decompilation Tool running in 5 minutes!

## Prerequisites

- Python 3.8+
- LLVM tools installed (`llc`, `clang`)
- A running model server (for local models) or API keys (for cloud models)

## 1. Install Dependencies

```bash
cd models/gemini
pip install -r requirements.txt
```

## 2. Setup Environment

### For Local Models (Qwen3-32B, Qwen3-30B-A3B)
Start your model server:
```bash
# Example for vLLM with Qwen3-32B
export CUDA_VISIBLE_DEVICES=0,1
vllm serve /path/to/Qwen3-32B --port 9001 --api-key token-llm4decompilation-abc123
```

### For Cloud Models
Set environment variables:
```bash
# For Huoshan-DeepSeek-R1
export ARK_STREAM_API_KEY="your_api_key"

# For OpenAI-GPT-4.1
export OPENAI_API_KEY="your_api_key"
```

### For RAG (Optional)
Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 3. Launch the Tool

### Option A: Using the launcher (Recommended)
```bash
python launch_tool.py
```

### Option B: Direct launch
```bash
python iterative_decompile.py
```

### Option C: With custom settings
```bash
python launch_tool.py --host 0.0.0.0 --port 7860 --share --debug
```

## 4. First Steps

1. **Open your browser** to `http://localhost:7860`

2. **Setup Model**:
   - Select your model from the dropdown
   - Configure host/port (for local models)
   - Click "Setup Model"

3. **Load Dataset**:
   - Enter dataset path (e.g., `~/Datasets/filtered_exebench/sampled_dataset_with_loops_164`)
   - Click "Load Dataset"

4. **Load Sample**:
   - Enter sample index (e.g., `0`)
   - Click "Load Sample"

5. **Generate Response**:
   - Review the generated prompt
   - Click "Generate Response"
   - Watch the streaming output!

## 5. Try Error Fixing

If the generated LLVM IR has errors:

1. **Check Evaluation Results**: Look at the JSON output
2. **Select Error Type**: Choose "compile_error" or "execution_error"
3. **Add Error Details**: Paste error messages or describe the issue
4. **Generate Fix**: Click "Generate Fix" to get an improved version

## Common Issues & Solutions

### "Model Connection Failed"
- Check if your model server is running
- Verify host/port settings
- For cloud models, check API keys

### "Dataset Loading Error"
- Verify dataset path exists
- Check dataset format matches expected structure
- Ensure you have read permissions

### "Qdrant Connection Error"
- Start Qdrant server: `docker run -p 6333:6333 qdrant/qdrant`
- Check host/port settings
- Disable in-context learning if not needed

### "Embedding Model Error"
- Verify embedding model path
- Check if model is compatible with sentence-transformers
- Disable in-context learning if not needed

## Example Workflow

1. **Load sample 0** from your dataset
2. **Review original code** in the tabs
3. **Generate LLVM IR** from assembly
4. **Check evaluation results**:
   - ✅ Green: Success!
   - ❌ Red: Needs fixing
5. **If errors exist**:
   - Copy error messages to "Error Message" field
   - Select appropriate error type
   - Generate fix
   - Repeat until success!

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Try the [demo.py](demo.py) script to see programmatic usage
- Customize prompts in `utils/openai_helper.py`
- Add your own evaluation metrics

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- Run with `--debug` flag for verbose output
- Check the browser console for JavaScript errors
- Verify all dependencies are installed correctly

Happy decompiling! 🚀


