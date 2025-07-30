import os
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from qdrant_client import QdrantClient

from models.rag.exebench_qdrant_base import load_embedding_model, build_qdrant_database, search_similar_records
from analysis.analyze_exebench_dataset import preprocessing_assembly, prepare_dataset


def build_exebench_train_synth_rich_io_qrand_database(dataset_dir: str = "/data1/xiachunwei/Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff"):
    # Load embedding model
    print("Loading embedding model...")
    model = load_embedding_model()
    # Connect to Qdrant
    client = QdrantClient("localhost", port=6333)
    for idx in range(8):
        # Load dataset
        sub_dataset_path = os.path.join(dataset_dir, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")
        collection_name = f"train_synth_rich_io_filtered_{idx}_preprocessed"
        print(f"Loading dataset from {sub_dataset_path}")
        dataset = load_from_disk(sub_dataset_path)
        print(f"Dataset loaded with {len(dataset)} records")
        client, collection_name = build_qdrant_database(dataset, client, model, collection_name, pre_process_asm_code=True)
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


def main_count_exebench_dataset_similarity():
    eval_data_path ='/data1/xiachunwei/Datasets/slade_artifact_cgo24_second/eval_data/exebench'
    num_proc = 80
    valid_synth = load_dataset(eval_data_path, split='valid_synth', trust_remote_code=True)
    test_synth = load_dataset(eval_data_path, split='test_synth', trust_remote_code=True)
    valid_synth = prepare_dataset(valid_synth, num_proc)
    test_synth = prepare_dataset(test_synth, num_proc)
    client = QdrantClient("localhost", port=6333)
    model = load_embedding_model()
    valid_synth_count = count_exebench_dataset_similarity(valid_synth, client, model)
    test_synth_count = count_exebench_dataset_similarity(test_synth, client, model)
    print(f"Valid synth count: {valid_synth_count}")
    print(f"Test synth count: {test_synth_count}")



if __name__ == "__main__":
    # main() 
    # test_build_database_and_search()
    # test_exebench_dataset_similarity()
    # test_query()
    # build_exebench_qrand_database()
    main_count_exebench_dataset_similarity()
