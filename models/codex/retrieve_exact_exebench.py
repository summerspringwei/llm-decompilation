"""Materialize exact Qdrant-retrieved ExeBench LLVM IR and C exemplars.

This evaluates retrieval benefit when a target assembly function has an exact
normalized-assembly counterpart in the indexed training shards.  It never reads
the target record's LLVM IR or C definition.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests
from datasets import load_from_disk
from qdrant_client import QdrantClient
from tqdm import tqdm

from config import HOME_DIR
from utils.preprocessing_assembly import preprocessing_assembly


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_with_loops_164"),
    "sampled_dataset_without_loops_164": os.path.join(HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_without_loops_164"),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164"),
}
TRAIN_ROOT = os.path.join(HOME_DIR, "Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve exact ExeBench assembly matches from Qdrant.")
    parser.add_argument("--dataset_name", required=True, choices=DATASET_PATHS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_url", default="http://localhost:8001/embed/batch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--collection_template", default="train_synth_rich_io_filtered_{idx}_preprocessed")
    return parser.parse_args()


def symbol_normalized(asm: str, fname: str) -> str:
    return re.sub(rf"\b{re.escape(fname)}\b", "__TARGET_FUNCTION__", asm)


def renamed(text: str, source_name: str, target_name: str) -> str:
    return re.sub(rf"\b{re.escape(source_name)}\b", target_name, text)


def force_emit_c_function(c_code: str, fname: str) -> str:
    pattern = rf"(?m)^(\s*(?:(?:static|inline|extern)\s+)*[^\n;{{}}]*\b{re.escape(fname)}\s*\()"

    def replace(match: re.Match[str]) -> str:
        # The C evaluator locates the function by its assembly symbol; static
        # functions have a local symbol, so make only the evaluation copy global.
        declaration = re.sub(r"\bstatic\s+", "", match.group(1), count=1)
        return f"__attribute__((used,noinline)) {declaration}"

    return re.sub(pattern, replace, c_code, count=1)


def embed_all(texts: list[str], url: str, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding targets", unit="batch"):
        response = requests.post(url, json=texts[start : start + batch_size], timeout=300)
        response.raise_for_status()
        vectors.extend(response.json()["embeddings"])
    return vectors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    llvm_dir = output_dir / "llvm_predictions"
    c_dir = output_dir / "c_predictions"
    metadata_dir = output_dir / "retrieval"
    for directory in (llvm_dir, c_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target = load_from_disk(DATASET_PATHS[args.dataset_name])
    train = [load_from_disk(os.path.join(TRAIN_ROOT, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff")) for idx in range(8)]
    target_asm = [preprocessing_assembly(record["asm"]["code"][-1], remove_comments=True) for record in target]
    vectors = embed_all(target_asm, args.embedding_url, args.batch_size)
    qdrant = QdrantClient("localhost", port=6333)

    exact_count = 0
    non_exact: list[int] = []
    scores: list[float] = []
    for target_idx, (record, asm, vector) in enumerate(tqdm(zip(target, target_asm, vectors), total=len(target), desc="Retrieving matches", unit="sample")):
        best_shard = -1
        best = None
        for shard in range(8):
            hit = qdrant.search(collection_name=args.collection_template.format(idx=shard), query_vector=vector, limit=1)[0]
            if best is None or hit.score > best.score:
                best_shard, best = shard, hit
        source_idx = int(best.payload["id"])
        source = train[best_shard][source_idx]
        source_asm = preprocessing_assembly(source["asm"]["code"][-1], remove_comments=True)
        exact = symbol_normalized(asm, record["fname"]) == symbol_normalized(source_asm, source["fname"])
        scores.append(float(best.score))
        metadata = {
            "score": best.score,
            "collection": args.collection_template.format(idx=best_shard),
            "point_id": best.id,
            "dataset_index": source_idx,
            "path": best.payload.get("path", ""),
            "source_fname": source["fname"],
            "target_fname": record["fname"],
            "normalized_assembly_exact_match": exact,
        }
        (metadata_dir / f"sample_{target_idx}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not exact:
            non_exact.append(target_idx)
            continue
        (llvm_dir / f"sample_{target_idx}.ll").write_text(renamed(source["llvm_ir"]["code"][-1], source["fname"], record["fname"]))
        c_code = renamed(source["func_def"], source["fname"], record["fname"])
        (c_dir / f"sample_{target_idx}.c").write_text(force_emit_c_function(c_code, record["fname"]))
        exact_count += 1

    summary = (
        f"total: {len(target)}\nexact_assembly_matches: {exact_count}\n"
        f"non_exact_matches: {len(non_exact)}\nnon_exact_indices: {non_exact}\n"
        f"score_min: {min(scores):.6f}\nscore_max: {max(scores):.6f}\n"
    )
    (output_dir / "retrieval_summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
