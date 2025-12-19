#!/usr/bin/env python3
import os
import torch
import logging
import numpy as np
from vllm import LLM
from tqdm import tqdm
from qdrant_client.models import Distance, VectorParams, PointStruct
from datasets import load_from_disk
from hermes

from utils.preprocessing_assembly import preprocessing_assembly


def build_qdrant_database(dataset, client, model, collection_name="assembly_code", batch_size = 64, pre_process_asm_code=True,
    vector_size = 4096  # Qwen3-Embedding-8B output size
):
    """Build Qdrant database from the dataset."""
    # Create collection
    
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
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        print(f"Inserted {i+batch_size} points")
    
    print(f"Successfully uploaded {len(points)} points to Qdrant collection: {collection_name}")
    return client, collection_name


def search_similar_records(client, collection_name, embedding_client, query_record, top_k=3, pre_process_asm_code=True):
    """Search for similar records in the database."""

    # Extract assembly code from query record
    if "asm" in query_record and "code" in query_record["asm"] and len(query_record["asm"]["code"]) > 0:
        query_assembly = query_record["asm"]["code"][-1]
        if pre_process_asm_code:
            query_assembly = preprocessing_assembly(query_assembly, remove_comments=True)
    else:
        raise ValueError("Query record does not contain assembly code")
    
    # Generate embedding for query
    query_embedding = embedding_client.embeddings.create(input = [query_assembly], model="Qwen3-Embedding-8B", encoding_format='float').data[0].embedding
    
    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )

    return search_results


def find_similar_records_in_exebench_synth_rich_io(client, model, record, dataset_dir):
    exebench_slits = [load_from_disk(os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")) for idx in range(8)]
    best_dataset_idx = 0
    best_record_idx = 0
    search_results = [(search_similar_records(client, f"train_synth_rich_io_filtered_{idx}_preprocessed", model, record, top_k=1), idx) for idx in range(8)] # list[list[tuple[search_result, idx]]]
    search_results.sort(key=lambda x: x[0][0].score, reverse=True)
    # We choose the second best result
    best_dataset_idx = search_results[1][1]
    best_record_idx = search_results[1][0][0].payload.get('id')
    best_score = search_results[1][0][0].score
    return (exebench_slits[best_dataset_idx][best_record_idx], best_score)
    