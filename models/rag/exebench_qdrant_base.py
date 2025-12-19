#!/usr/bin/env python3
import os
import torch
import logging
import requests
import numpy as np
from vllm import LLM
from tqdm import tqdm
from qdrant_client.models import Distance, VectorParams, PointStruct
from datasets import load_from_disk
from qdrant_client import QdrantClient
from utils.preprocessing_assembly import preprocessing_assembly


def get_embeddings(texts, embedding_client, batch_size=64) -> list[np.ndarray]:
    """Generate embeddings for a list of texts using vLLM."""
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size),
                  desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Generate embeddings using vLLM
        outputs = embedding_client.embeddings.create(
            input=batch_texts,
            model="Qwen3-Embedding-8B",
            encoding_format='float')
        batch_embeddings = torch.tensor([o.embedding for o in outputs.data])
        embeddings.extend(batch_embeddings)

    return embeddings


def get_exebench_hermessim_embedding_batch(url,
                                           texts,
                                           batch_size=64) -> list[np.ndarray]:
    embeddings = []
    success_indices = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        response = requests.post(url, json=batch_texts)
        response.raise_for_status()
        results = response.json()
        batch_embeddings = results['embeddings']
        batch_success_indices = [i + j for j in results['success_indices']]
        embeddings.extend(batch_embeddings)
        success_indices.extend(batch_success_indices)

    return embeddings, success_indices


def build_qdrant_database(
        dataset,
        client,
        collection_name="assembly_code",
        batch_size=64,
        pre_process_asm_code=True,
        vector_size=4096  # Qwen3-Embedding-8B output size
):
    """Build Qdrant database from the dataset."""
    # Create collection

    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    client.create_collection(collection_name=collection_name,
                             vectors_config=VectorParams(
                                 size=vector_size, distance=Distance.COSINE))
    print(f"Created collection: {collection_name}")

    # Extract assembly code and metadata
    assembly_texts = []
    metadata_list = []

    print("Extracting assembly code...")
    for i, record in enumerate(tqdm(dataset, desc="Processing records")):
        try:
            # Get the last assembly code from record["asm"]["code"][-1]
            if "asm" in record and "code" in record["asm"] and len(
                    record["asm"]["code"]) > 0:
                assembly_code = record["asm"]["code"][-1]
                logging.debug(assembly_code)
                logging.debug('-' * 50)
                if pre_process_asm_code:
                    assembly_code = preprocessing_assembly(
                        assembly_code, remove_comments=True)
                    logging.debug(assembly_code)
                    logging.debug('=' * 50)
                # Prepare metadata
                metadata = {"id": i, "path": record.get("path", "")}

                assembly_texts.append(assembly_code)
                metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue

    print(f"Extracted {len(assembly_texts)} assembly code samples")

    # Generate embeddings
    print("Generating embeddings...")
    # embeddings = get_embeddings(assembly_texts, model, batch_size)
    url = "http://localhost:8123/embed/batch"
    embeddings, success_indices = get_exebench_hermessim_embedding_batch(
        url, assembly_texts, batch_size)

    # Prepare points for Qdrant
    points = []
    for i, (embedding, idx) in enumerate(zip(embeddings, success_indices)):
        point = PointStruct(id=i, vector=embedding, payload=metadata_list[idx])
        points.append(point)
    # Upload to Qdrant
    print("Uploading to Qdrant...")

    for i in range(0, len(points), batch_size):
        batch_points = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch_points)
        print(f"Inserted {i+batch_size} points")

    print(
        f"Successfully uploaded {len(points)} points to Qdrant collection: {collection_name}"
    )
    return client, collection_name


def search_similar_records(client,
                           collection_name,
                           query_embedding,
                           top_k=3):
    """Search for similar records in the database."""

    # Search in Qdrant
    search_results = client.search(collection_name=collection_name,
                                   query_vector=query_embedding,
                                   limit=top_k)

    return search_results


class ExebenchQdrantSearch:
    def __init__(self, dataset_dir: str, client: QdrantClient, embedding_url: str, collection_name_with_idx: str):
        self.exebench_slits = [
            load_from_disk(
                os.path.join(
                    dataset_dir,
                    f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff"
                )) for idx in range(8)
        ]
        self.client = client
        self.embedding_url = embedding_url
        self.collection_name_with_idx = collection_name_with_idx
    
    def find_similar_records_in_exebench_synth_rich_io(self, query_record: dict, pre_process_asm_code: bool = True):
        best_dataset_idx = 0
        best_record_idx = 0
        # Get the embedding of the record
        # Extract assembly code from query record
        if "asm" in query_record and "code" in query_record["asm"] and len(
                query_record["asm"]["code"]) > 0:
            query_assembly = query_record["asm"]["code"][-1]
            if pre_process_asm_code:
                query_assembly = preprocessing_assembly(query_assembly,
                                                        remove_comments=True)
        else:
            raise ValueError("Query record does not contain assembly code")
        response = requests.post(self.embedding_url, json=[query_assembly])
        response.raise_for_status()
        results = response.json()
        query_embedding = results['embeddings'][0]
        # query_embedding = requests.post(self.embedding_url, json=query_assembly).json()['embeddings']

        search_results = [(search_similar_records(
            self.client,
            self.collection_name_with_idx.format(idx=idx),
            query_embedding,
            top_k=1), idx) for idx in range(8)
        ]  # list[list[tuple[search_result, idx]]]

        search_results.sort(key=lambda x: x[0][0].score, reverse=True)
        best_dataset_idx = search_results[1][1]
        best_record_idx = search_results[1][0][0].payload.get('id')
        best_score = search_results[1][0][0].score
        return (self.exebench_slits[best_dataset_idx][best_record_idx], best_score)

