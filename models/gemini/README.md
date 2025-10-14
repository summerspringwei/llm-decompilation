# Interactive Decompilation Tool

A Gradio-based interactive tool for iterative decompilation of assembly code to LLVM IR using large language models.

## Features

- **Dataset Loading**: Load datasets from disk and browse through samples
- **Model Configuration**: Support for multiple LLM models (Qwen3-32B, Qwen3-30B-A3B, Huoshan-DeepSeek-R1, OpenAI-GPT-4.1)
- **RAG Integration**: In-context learning with similar examples from Qdrant vector database
- **Streaming Generation**: Real-time response generation with streaming
- **Interactive Evaluation**: Automatic compilation and execution testing of generated LLVM IR
- **Error Fixing**: Iterative error correction with compile and execution error feedback
- **Visual Interface**: Clean, organized interface with tabs for different code views

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Set up the following environment variables for API access:

```bash
export ARK_STREAM_API_KEY="your_ark_stream_api_key"  # For Huoshan-DeepSeek-R1
export OPENAI_API_KEY="your_openai_api_key"         # For OpenAI-GPT-4.1
```

### 3. Model Server Setup

For local models (Qwen3-32B, Qwen3-30B-A3B), ensure your model server is running:

```bash
export CUDA_VISIBLE_DEVICES=0,1 && vllm serve /data1/xiachunwei/Datasets/Models/Qwen3-32B --port 9001 --api-key token-llm4decompilation-abc123 --tensor-parallel-size 2 --served-model-name Qwen3-32B
```


### 4. Qdrant Setup

For RAG functionality, ensure Qdrant is running:

```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant
```

Start embedding model service:
```bash
python3 models/rag/embedding_service.py
```
## Usage

### 1. Start the Tool

```bash
cd models/gemini
python iterative_decompile.py
```

The tool will be available at `http://localhost:7860`

### 2. Configuration Steps

1. **Setup Model**: Configure your chosen model with appropriate host/port settings
2. **Load Dataset**: Provide the path to your dataset directory
3. **Select Sample**: Choose a sample index to work with

### 3. Workflow

1. **Load Sample**: Click "Load Sample" to load a specific sample by index
2. **Review Original Code**: Examine the original C function, assembly, and LLVM IR
3. **Generate Response**: Click "Generate Response" to create LLVM IR from assembly
4. **Evaluate Results**: Review compilation and execution success
5. **Fix Errors**: If needed, use the error fixing section to iteratively improve results

### 4. Error Fixing

The tool supports two types of error fixing:

- **Compile Error**: When the generated LLVM IR fails to compile
- **Execution Error**: When the compiled assembly doesn't match expected behavior

For each error type, the tool automatically generates appropriate prompts with error messages and context.

## Interface Components

### Model Configuration
- Model selection dropdown
- Host/port configuration for local models
- Qdrant connection settings
- Embedding model path for RAG
- In-context learning toggle

### Dataset Management
- Dataset path input
- Sample index selection
- Dataset status display

### Code Display
- **Original Code Tab**: C function definition, original assembly, original LLVM IR
- **Similar Record Tab**: RAG-retrieved similar examples
- **Predicted Code Tab**: Generated LLVM IR and compiled assembly

### Generation Controls
- Prompt display and editing
- Number of generations slider
- Generate button with streaming output

### Evaluation Results
- JSON display of compilation and execution results
- Success/failure indicators
- Error messages

### Error Fixing
- Error type selection (compile/execution)
- Error message input
- Automatic fix prompt generation
- Fixed response evaluation

## Dataset Format

The tool expects datasets in the following format:

```python
{
    "func_head": "C function definition",
    "asm": {"code": ["assembly_code"]},
    "llvm_ir": {"code": ["llvm_ir_code"]},
    # ... other fields
}
```

## Model Support

### Local Models
- **Qwen3-32B**: Requires local server on specified host/port
- **Qwen3-30B-A3B**: Requires local server on specified host/port

### Cloud Models
- **Huoshan-DeepSeek-R1**: Requires ARK_STREAM_API_KEY
- **OpenAI-GPT-4.1**: Requires OPENAI_API_KEY

## RAG Configuration

For in-context learning, the tool uses:
- Qdrant vector database for similarity search
- Sentence transformers for embeddings
- Similar assembly/LLVM IR pairs as examples

## Troubleshooting

### Common Issues

1. **Model Connection Failed**: Check host/port settings and ensure model server is running
2. **Dataset Loading Error**: Verify dataset path and format
3. **Qdrant Connection Error**: Ensure Qdrant server is running on specified host/port
4. **Embedding Model Error**: Check embedding model path and ensure it's compatible

### Debug Mode

Enable debug logging by modifying the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Advanced Usage

### Custom Prompts

You can modify the prompt templates in `utils/openai_helper.py`:
- `GENERAL_INIT_PROMPT`: Basic decompilation prompt
- `SIMILAR_RECORD_PROMPT`: In-context learning prompt
- `format_compile_error_prompt`: Compile error fixing prompt
- `format_execution_error_prompt`: Execution error fixing prompt

### Custom Evaluation

The evaluation logic can be customized in the `evaluate_response` method to add additional validation criteria.

### Batch Processing

For batch processing, you can extend the tool to process multiple samples automatically by modifying the event handlers.

## Contributing

To extend the tool:

1. Add new models to `SERVICE_CONFIG`
2. Implement new evaluation metrics
3. Add custom prompt templates
4. Extend the UI with new components

## License

This tool is part of the LLM decompilation project. Please refer to the main project license.