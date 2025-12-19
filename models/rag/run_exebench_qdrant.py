import os
import requests
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from qdrant_client import QdrantClient

from models.rag.exebench_qdrant_base import build_qdrant_database, search_similar_records
from utils.preprocessing_assembly import preprocessing_assembly


def build_exebench_train_synth_rich_io_qrand_database(dataset_dir: str = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff",
    collection_name_suffix: str = "hermessim",
    vector_size: int = 384):
    # Connect to Qdrant
    client = QdrantClient("localhost", port=6333)
    for idx in range(3, 8):
        # Load dataset
        sub_dataset_path = os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")
        collection_name = f"train_synth_rich_io_filtered_{idx}_preprocessed_{collection_name_suffix}"
        print(f"Loading dataset from {sub_dataset_path}")
        dataset = load_from_disk(sub_dataset_path)
        print(f"Dataset loaded with {len(dataset)} records")
        client, collection_name = build_qdrant_database(dataset, client, collection_name, batch_size = 64, pre_process_asm_code=True, vector_size=vector_size)
        print(f"Built database for {collection_name}")



def count_exebench_dataset_similarity(dataset, client, model):
    count = 0
    dataset_dir: str = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"
    exebench_slits = [load_from_disk(os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")) for idx in range(8)]
    for record in tqdm(dataset, desc="Counting exebench dataset similarity"):
        found = False
        print("="*50)
        print(preprocessing_assembly(record["asm"]["code"][-1]))
        for idx in range(8):
            collection_name = f"train_synth_rich_io_filtered_{idx}_preprocessed"
            search_results = search_similar_records(client, collection_name, model, record, top_k=3)
            for i, result in enumerate(search_results, 1):
                similar_idx = result.payload.get('id')
                print(f"Similar record: idx: {similar_idx} {result.payload.get('path', 'N/A')}")
                print(f"Similarity score: {result.score:.4f}")
                print(preprocessing_assembly(exebench_slits[idx][similar_idx]["asm"]["code"][-1]))
                if result.score > 0.9:
                    count += 1
                    found = True
                    break
            if found:
                break
    return count

def find_missing_indices_in_collection(client, collection_name, dataset_count):
    """
    Query all the points in the collection, return their ids, 
    then iterate from 0 to count to see which idx is missing.
    """
    offset = None
    all_ids = []
    while True:
        # 1. Fetch a page of points
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=100,           # Adjust batch size based on memory
            with_payload=True,   # Set to False if you only need IDs
            with_vectors=False,  # Set to True if you need the embeddings
            offset=offset,
        )

        # 2. Process your records
        for record in records:
            all_ids.append(record.payload.get('id'))
        # 3. Update offset for next page
        offset = next_page_offset

        # 4. Stop when no more points are left
        if offset is None:
            break
    
    for idx in range(dataset_count):
        if idx not in all_ids:
            print(f"Missing index: {idx}")


def count_exebench_failed_records():
    client = QdrantClient("localhost", port=6333)
    dataset_dir: str = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"
    exebench_slits = [load_from_disk(os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")) for idx in range(8)]
    for idx in range(8):
        collection_name = f"train_synth_rich_io_filtered_{idx}_preprocessed_hermessim"
        count = client.count(collection_name=collection_name)
        dataset_count = len(exebench_slits[idx])
        print(f"Collection {collection_name} has {count} records, dataset has {dataset_count} records")
        if count.count != int(dataset_count):
            print(f"Difference: Collection {collection_name} has {count} records, dataset has {dataset_count} records")
            find_missing_indices_in_collection(client, collection_name, dataset_count)


def find_similar_records_in_exebench_synth_rich_io():
    client = QdrantClient("localhost", port=6333)
    dataset_dir: str = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"
    exebench_100 = load_from_disk(os.path.join(dataset_dir, "train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff_sample_100"))
    exebench_slits = [load_from_disk(os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")) for idx in range(8)]
    url = "http://localhost:8123/embed/"
    for query_record in exebench_100:
        # Extract assembly code from query record
        if "asm" in query_record and "code" in query_record["asm"] and len(
                query_record["asm"]["code"]) > 0:
            query_assembly = query_record["asm"]["code"][-1]
            query_assembly = preprocessing_assembly(query_assembly,
                                                    remove_comments=True)
        else:
            raise ValueError("Query record does not contain assembly code")
        print(f"```C\n{query_record['func_def']}\n```")
        response = requests.post(url, json=query_assembly)
        response.raise_for_status()
        query_embedding = response.json()["embedding"]
        for idx in range(1, 8):
            collection_name = f"train_synth_rich_io_filtered_{idx}_preprocessed_hermessim"
            search_results = client.search(collection_name=collection_name, query_vector=query_embedding[0], limit=2)
            for i, result in enumerate(search_results, 1):
                similar_idx = result.payload.get('id')
                print(f"Similar record: idx: {similar_idx} {result.payload.get('path', 'N/A')}")
                print(f"Similarity score: {result.score:.4f}")
                print(f"```C\n{exebench_slits[idx][similar_idx]['func_def']}\n```")
        print("="*50)



# def main_count_exebench_dataset_similarity():
#     eval_data_path ='/data1/xiachunwei/Datasets/slade_artifact_cgo24_second/eval_data/exebench'
#     num_proc = 80
#     valid_synth = load_dataset(eval_data_path, split='valid_synth', trust_remote_code=True)
#     test_synth = load_dataset(eval_data_path, split='test_synth', trust_remote_code=True)
#     valid_synth = prepare_dataset(valid_synth, num_proc)
#     test_synth = prepare_dataset(test_synth, num_proc)
#     client = QdrantClient("localhost", port=6333)
#     model = load_embedding_model()
#     valid_synth_count = count_exebench_dataset_similarity(valid_synth, client, model)
#     test_synth_count = count_exebench_dataset_similarity(test_synth, client, model)
#     print(f"Valid synth count: {valid_synth_count}")
#     print(f"Test synth count: {test_synth_count}")



if __name__ == "__main__":
    # main() 
    # test_build_database_and_search()
    # test_exebench_dataset_similarity()
    # test_query()
    # build_exebench_qrand_database()
    # main_count_exebench_dataset_similarity()
    # build_exebench_train_synth_rich_io_qrand_database()
    # count_exebench_failed_records()
    find_similar_records_in_exebench_synth_rich_io()
