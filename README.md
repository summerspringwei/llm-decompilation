# Qdrant Assembly Code Database

This project builds a Qdrant vector database for assembly code similarity search using the Qwen/Qwen3-Embedding-8B model with vLLM for high-performance embedding generation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Qdrant is running on localhost:6333:
```bash
# If you need to start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### 1. Build the Database

Run the main script to build the Qdrant database from the dataset:

```bash
python build_qdrant_database.py
```

This script will:
- Load the Arrow dataset from the specified path
- Extract assembly code from `record["asm"]["code"][-1]`
- Generate embeddings using Qwen/Qwen3-Embedding-8B via vLLM (much faster than transformers)
- Upload vectors and metadata to Qdrant

### 2. Test Search

Test the similarity search functionality using the provided record.json:

```bash
python test_qdrant_search.py
```

This will:
- Load the test record from `tests/record.json`
- Extract its assembly code
- Search for the top 3 most similar records in the database
- Display results with similarity scores

### 3. Check Database Status

Check the status of your Qdrant database:

```bash
python check_qdrant_status.py
```

## Dataset Structure

The dataset contains records with the following structure:
- `asm.code[-1]`: Assembly code (used for embeddings)
- `path`: File path
- `fname`: Function name
- `func_head`: Function signature
- `asm.target`: Compiler target

## Search Results

The search returns:
- Similarity score (cosine similarity)
- Record ID
- File path and function name
- Assembly code preview
- Compiler target information

## Configuration

- **Embedding Model**: Qwen/Qwen3-Embedding-8B (4096-dimensional vectors)
- **Embedding Engine**: vLLM (high-performance inference)
- **Distance Metric**: Cosine similarity
- **Collection Name**: "assembly_code"
- **Qdrant Port**: 6333
- **Model Path**: `/home/xiachunwei/Datasets/Models/Qwen3-Embedding-8B`

## Performance Benefits

- **vLLM Integration**: Much faster embedding generation compared to transformers
- **Batch Processing**: Efficient batch processing with larger batch sizes (32 vs 8)
- **GPU Optimization**: Automatic GPU acceleration with vLLM
- **Memory Efficient**: Better memory management for large datasets

## Notes

- The script processes assembly code in batches for efficiency
- vLLM provides significant speedup for embedding generation
- GPU acceleration is automatically used if available
- Assembly code is processed directly without tokenization limits
- Metadata includes relevant information for each assembly code sample
