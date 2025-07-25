#!/usr/bin/env python3

import torch
from vllm import LLM
from tqdm import tqdm
from datasets import load_from_disk
from qdrant_client import QdrantClient  
from qdrant_client.models import Distance, VectorParams, PointStruct
import os

def setup_cuda_environment():
    """Setup CUDA environment to avoid MPS conflicts."""
    # Force CUDA usage and disable MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Check if CUDA is actually available
    try:
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s)")
            print(f"Current device: {torch.cuda.current_device()}")
            return True
        else:
            print("CUDA is not available, will use CPU")
            return False
    except Exception as e:
        print(f"CUDA check failed: {e}")
        print("Will use CPU for embedding generation")
        return False

def load_embedding_model():
    """Load the Qwen/Qwen3-Embedding-8B model using vLLM for generating embeddings."""
    # Setup CUDA environment
    cuda_available = setup_cuda_environment()
    
    if cuda_available:
        # Use GPU with vLLM
        model = LLM(
            model="/home/xiachunwei/Datasets/Models/Qwen3-Embedding-8B", 
            task="embed",
            gpu_memory_utilization=0.8,
            max_model_len=512
        )
    else:
        # Use CPU with vLLM
        model = LLM(
            model="/home/xiachunwei/Datasets/Models/Qwen3-Embedding-8B", 
            task="embed",
            gpu_memory_utilization=0.0,  # Force CPU usage
            max_model_len=512
        )
    
    return model

def get_embeddings(texts, model, batch_size=16):
    """Generate embeddings for a list of texts using vLLM."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Generate embeddings using vLLM
        outputs = model.embed(batch_texts)
        batch_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        embeddings.extend(batch_embeddings.cpu().numpy())
    
    return embeddings

def build_qdrant_database(dataset_path, collection_name="assembly_code"):
    """Build Qdrant database from the dataset."""
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} records")
    
    # Load embedding model
    print("Loading embedding model...")
    model = load_embedding_model()
    
    # Connect to Qdrant
    client = QdrantClient("localhost", port=6333)
    
    # Create collection
    vector_size = 4096  # Qwen3-Embedding-8B output size
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection_name}")
    
    # Extract assembly code and metadata
    assembly_texts = []
    metadata_list = []
    
    print("Extracting assembly code...")
    for i, record in enumerate(tqdm(dataset, desc="Processing records")):
        try:
            # Get the last assembly code from record["asm"]["code"][-1]
            if "asm" in record and "code" in record["asm"] and len(record["asm"]["code"]) > 0:
                assembly_code = record["asm"]["code"][-1]
                
                # Prepare metadata
                metadata = {
                    "id": i,
                    "path": record.get("path", ""),
                    # "fname": record.get("fname", ""),
                    # "func_head": record.get("func_head", ""),
                    # "target": record["asm"].get("target", [])[-1] if record["asm"].get("target") else "",
                    # "assembly_code": assembly_code[:1000] + "..." if len(assembly_code) > 1000 else assembly_code
                }
                
                assembly_texts.append(assembly_code)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue
    
    print(f"Extracted {len(assembly_texts)} assembly code samples")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(assembly_texts, model)
    
    # Prepare points for Qdrant
    points = []
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=metadata
        )
        points.append(point)
    
    # Upload to Qdrant
    print("Uploading to Qdrant...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully uploaded {len(points)} points to Qdrant collection: {collection_name}")
    return client, collection_name

def search_similar_records(client, collection_name, query_record, top_k=3):
    """Search for similar records in the database."""
    
    # Load embedding model
    model = load_embedding_model()
    
    # Extract assembly code from query record
    if "asm" in query_record and "code" in query_record["asm"] and len(query_record["asm"]["code"]) > 0:
        query_assembly = query_record["asm"]["code"][-1]
    else:
        raise ValueError("Query record does not contain assembly code")
    
    # Generate embedding for query
    outputs = model.embed([query_assembly])
    query_embedding = torch.tensor([o.outputs.embedding for o in outputs])[0].cpu().numpy()
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    return search_results

def main():
    # Dataset path
    dataset_path = "/home/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"
    
    # Build database
    client, collection_name = build_qdrant_database(dataset_path)
    
    # Load a sample record for testing
    dataset = load_from_disk(dataset_path)
    sample_record = dataset[0]  # Use first record as query
    
    print("\n" + "="*50)
    print("SAMPLE QUERY RECORD:")
    print(f"Path: {sample_record.get('path', 'N/A')}")
    print(f"Function: {sample_record.get('fname', 'N/A')}")
    print(f"Assembly code preview: {sample_record['asm']['code'][-1][:200]}...")
    
    # Search for similar records
    print("\n" + "="*50)
    print("SEARCHING FOR SIMILAR RECORDS:")
    search_results = search_similar_records(client, collection_name, sample_record, top_k=3)
    
    for i, result in enumerate(search_results, 1):
        print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
        print(f"ID: {result.id}")
        print(f"Path: {result.payload.get('path', 'N/A')}")
        print(f"Function: {result.payload.get('fname', 'N/A')}")
        print(f"Target: {result.payload.get('target', 'N/A')}")
        print(f"Assembly code preview: {result.payload.get('assembly_code', 'N/A')[:200]}...")

if __name__ == "__main__":
    main() 