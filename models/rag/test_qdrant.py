from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from qdrant_client import QdrantClient
from models.rag.build_qdrant_database import build_qdrant_database, search_similar_records, load_embedding_model, find_similar_records_in_exebench_synth_rich_io
from utils.preprocessing_assembly import preprocessing_assembly 
from analysis.analyze_exebench_dataset import prepare_dataset

import logging
logging.basicConfig(level=logging.INFO)


def test_build_database_and_search():
    # Dataset path
    dataset_path = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"
    collection_name = "assembly_code"
    # Build database
    client, collection_name = build_qdrant_database(dataset_path, collection_name)
    # client = QdrantClient("localhost", port=6333)
    # Load a sample record for testing
    dataset = load_from_disk(dataset_path)
    sample_record = dataset[1000]  # Use first record as query
    
    print("\n" + "="*50)
    print("SAMPLE QUERY RECORD:")
    print(f"Path: {sample_record.get('path', 'N/A')}")
    print(f"Function: {sample_record.get('fname', 'N/A')}")
    print(f"Assembly code preview: {sample_record['asm']['code'][-1][:200]}...")
    
    # Search for similar records
    print("\n" + "="*50)
    print("SEARCHING FOR SIMILAR RECORDS:")
    search_results = search_similar_records(client, collection_name, sample_record, top_k=3)
    print(sample_record["asm"]["code"][-1])
    for i, result in enumerate(search_results, 1):
        print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
        print(f"ID: {result.id}")
        print(f"Path: {result.payload.get('path', 'N/A')}")
        record = dataset[result.id]
        print((record["asm"]["code"][-1]))
        # print(f"Function: {result.payload.get('fname', 'N/A')}")
        # print(f"Target: {result.payload.get('target', 'N/A')}")
        # print(f"Assembly code preview: {result.payload.get('assembly_code', 'N/A')[:200]}...")


def test_query():
    client = QdrantClient("localhost", port=6333)
    path = "/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    sample_record = load_from_disk(path).select(range(3))

    query_dataset_path = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff/train_synth_rich_io_filtered_0_llvm_extract_func_ir_assembly_O2_llvm_diff"
    query_dataset = load_from_disk(query_dataset_path)
    # Load embedding model
    model = load_embedding_model()
    for record in sample_record:
        search_results = search_similar_records(client, "assembly_code", model, record, top_k=1)
        print("="*50)
        print(record["path"])
        print(preprocessing_assembly(record["asm"]["code"][-1]))
        for i, result in enumerate(search_results, 1):
            print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
            print(f"ID: {result.id}")
            print(f"Path: {result.payload.get('path', 'N/A')}")
            query_record = query_dataset[result.id]
            print(preprocessing_assembly(query_record["asm"]["code"][-1]))


def create_exebench_test_validation_dataset():
    client = QdrantClient("localhost", port=6333)
    eval_data_path ='/data1/xiachunwei/Datasets/slade_artifact_cgo24_second/eval_data/exebench'
    num_proc = 80
    valid_synth = load_dataset(eval_data_path, split='valid_synth', trust_remote_code=True)
    test_synth = load_dataset(eval_data_path, split='test_synth', trust_remote_code=True)
    valid_synth = prepare_dataset(valid_synth, num_proc)
    test_synth = prepare_dataset(test_synth, num_proc)
    client = QdrantClient("localhost", port=6333)
    model = load_embedding_model()
    build_qdrant_database(valid_synth, client, model, collection_name='exebench_valid_synth')
    build_qdrant_database(test_synth, client, model, collection_name='exebench_test_synth')
    print("Exebench test and validation dataset created")


def verify_self_similarity():
    client = QdrantClient("localhost", port=6333)
    eval_data_path ='/data1/xiachunwei/Datasets/slade_artifact_cgo24_second/eval_data/exebench'
    num_proc = 80
    valid_synth = load_dataset(eval_data_path, split='valid_synth', trust_remote_code=True)
    test_synth = load_dataset(eval_data_path, split='test_synth', trust_remote_code=True)
    valid_synth = prepare_dataset(valid_synth, num_proc)
    test_synth = prepare_dataset(test_synth, num_proc)
    # Load embedding model
    print("Loading embedding model...")
    model = load_embedding_model()
    count = 0
    for record in tqdm(valid_synth, desc="Verifying self similarity"):
        search_results = search_similar_records(client, 'exebench_valid_synth', model, record, top_k=1)
        for i, result in enumerate(search_results, 1):
            print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
            if result.score > 0.95:
                count += 1
    print(f"Total self similarity count: {count}")


def verify_hard_case():
    dataset_path = "/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    dataset = load_from_disk(dataset_path)
    client = QdrantClient("localhost", port=6333)
    model = load_embedding_model()
    count = 0
    for record in tqdm(dataset, desc="Verifying hard case"):
        search_results = search_similar_records(client, 'exebench_valid_synth', model, record, top_k=1)
        for i, result in enumerate(search_results, 1):
            print(f"\n--- valid_synth Result {i} (Score: {result.score:.4f}) ---")
            if result.score > 0.95:
                count += 1
        search_results = search_similar_records(client, 'exebench_test_synth', model, record, top_k=1)
        for i, result in enumerate(search_results, 1):
            print(f"\n--- test_synth Result {i} (Score: {result.score:.4f}) ---")
            if result.score > 0.95:
                count += 1


def find_same():
    dataset_path = "/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    dataset = load_from_disk(dataset_path)
    eval_data_path ='/data1/xiachunwei/Datasets/slade_artifact_cgo24_second/eval_data/exebench'
    num_proc = 80
    valid_synth = load_dataset(eval_data_path, split='valid_synth', trust_remote_code=True)
    test_synth = load_dataset(eval_data_path, split='test_synth', trust_remote_code=True)
    valid_synth = prepare_dataset(valid_synth, num_proc)
    test_synth = prepare_dataset(test_synth, num_proc)
    for record in dataset:
        for valid_record in valid_synth:
            if record["path"] == valid_record["path"] and record['func_head'] == valid_record['func_head']:
                print(f"Found same in valid_synth: {record['path']}")
                break
        for test_record in test_synth:
            if record["path"] == test_record["path"] and record['func_head'] == test_record['func_head']:
                print(f"Found same in test_synth: {record['path']}")



def find_similar_for_motivation_dataset():
    dataset_dir = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"
    dataset_path = "/data1/xiachunwei/Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    dataset = load_from_disk(dataset_path)
    client = QdrantClient("localhost", port=6333)
    model = load_embedding_model()
    for record in dataset:
        similar_records, score = find_similar_records_in_exebench_synth_rich_io(client, model, record, dataset_dir)
        print("//"*50)
        print(f"Similar record: {similar_records['path']}")
        print(f"Similarity score: {score:.4f}")
        print(record['func_def'])
        print("//"*50)
        print(similar_records['func_def'])
        
        


if __name__ == "__main__":
    # create_exebench_test_validation_dataset()
    # verify_self_similarity()
    # verify_hard_case()
    # find_same()
    find_similar_for_motivation_dataset()