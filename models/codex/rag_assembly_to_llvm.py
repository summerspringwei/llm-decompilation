"""Retrieve ExeBench exemplars from Qdrant and generate LLVM IR from assembly.

For every target function this runner embeds normalized assembly, retrieves the
nearest record across the eight Qwen3/Qdrant training shards, and supplies that
record's normalized assembly and LLVM IR as an in-context example to a local
OpenAI-compatible generation server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_from_disk
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

from config import HOME_DIR
from utils.preprocessing_assembly import preprocessing_assembly
from utils.preprocessing_llvm_ir import preprocessing_llvm_ir
from utils.prompt_templates import SIMILAR_RECORD_PROMPT


DATASET_PATHS = {
    "sampled_dataset_with_loops_164": os.path.join(
        HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_with_loops_164"
    ),
    "sampled_dataset_without_loops_164": os.path.join(
        HOME_DIR, "Datasets/filtered_exebench/sampled_dataset_without_loops_164"
    ),
    "sampled_dataset_with_loops_and_only_one_bb_164": os.path.join(
        HOME_DIR,
        "Datasets/filtered_exebench/sampled_dataset_with_loops_and_only_one_bb_164",
    ),
}

TRAIN_ROOT = os.path.join(
    HOME_DIR,
    "Datasets/filtered_exebench/train_synth_rich_io_filtered_llvm_extract_func_ir_assembly_O2_llvm_diff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-guided assembly-to-LLVM decompilation.")
    parser.add_argument("--dataset_name", required=True, choices=DATASET_PATHS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_url", default="http://localhost:8001/embed/batch")
    parser.add_argument("--qdrant_host", default="localhost")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    parser.add_argument("--collection_template", default="train_synth_rich_io_filtered_{idx}_preprocessed")
    parser.add_argument("--base_url", default="http://localhost:9002/v1")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--api_key", default="token-llm4decompilation-abc123")
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def selected_indices(size: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices:
        return [int(item) for item in args.sample_indices.split(",") if item]
    if args.max_samples:
        return list(range(min(size, args.max_samples)))
    return list(range(size))


def extract_llvm(text: str) -> str:
    match = re.search(r"```(?:llvm|LLVM)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    text = re.sub(r"^```(?:llvm|LLVM)?\s*\n", "", text.strip())
    module_start = text.find("; ModuleID")
    if module_start < 0:
        module_start = text.find("target datalayout")
    define_start = text.find("define ", module_start if module_start >= 0 else 0)
    if define_start < 0:
        return text.removesuffix("```").strip() + "\n"
    start = module_start if module_start >= 0 else define_start
    body_start = text.find("{", define_start)
    depth = 0
    for pos in range(body_start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                code = text[start : pos + 1]
                kept: list[str] = []
                in_function = False
                for line in code.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("define "):
                        kept.append(line)
                        in_function = True
                    elif in_function:
                        if line[:1].isspace() or stripped.endswith(":") or stripped == "}":
                            kept.append(line)
                    elif stripped.startswith((";", "source_", "target ", "%", "@", "declare ", "define ", "attributes ", "!")) or not stripped:
                        kept.append(line)
                return re.sub(r"\s+#\d+(?=\s*\{)", "", "\n".join(kept)) + "\n"
    return text[start:].removesuffix("```").strip() + "\n"


def embed_texts(url: str, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding targets", unit="batch"):
        response = requests.post(url, json=texts[start : start + batch_size], timeout=300)
        response.raise_for_status()
        batch = response.json()["embeddings"]
        if len(batch) != min(batch_size, len(texts) - start):
            raise RuntimeError("embedding service returned a mismatched batch size")
        vectors.extend(batch)
    return vectors


def retrieve_one(
    client: QdrantClient,
    vector: list[float],
    collection_template: str,
) -> tuple[int, object]:
    best_idx = -1
    best = None
    for shard_idx in range(8):
        results = client.search(
            collection_name=collection_template.format(idx=shard_idx),
            query_vector=vector,
            limit=1,
        )
        if results and (best is None or results[0].score > best.score):
            best_idx, best = shard_idx, results[0]
    if best is None:
        raise RuntimeError("no Qdrant result returned")
    return best_idx, best


def generate_one(client: OpenAI, model: str, prompt: str, max_tokens: int) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    message = response.choices[0].message
    raw = (getattr(message, "reasoning_content", None) or "") + "\n\n" + (message.content or "")
    return extract_llvm(raw), raw


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    prompts_dir = output_dir / "prompts"
    retrieval_dir = output_dir / "retrieval"
    for path in (predictions_dir, prompts_dir, retrieval_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(DATASET_PATHS[args.dataset_name])
    indices = selected_indices(len(dataset), args)
    targets = [dataset[idx] for idx in indices]
    target_asm = [preprocessing_assembly(record["asm"]["code"][-1], remove_comments=True) for record in targets]
    vectors = embed_texts(args.embedding_url, target_asm, args.embedding_batch_size)

    qdrant = QdrantClient(args.qdrant_host, port=args.qdrant_port)
    train_sets = [
        load_from_disk(os.path.join(TRAIN_ROOT, f"train_synth_rich_io_filtered_{idx}_llvm_extract_func_ir_assembly_O2_llvm_diff"))
        for idx in range(8)
    ]
    jobs = []
    for idx, asm, vector in tqdm(zip(indices, target_asm, vectors), total=len(indices), desc="Retrieving exemplars", unit="sample"):
        shard_idx, hit = retrieve_one(qdrant, vector, args.collection_template)
        source_idx = int(hit.payload["id"])
        source = train_sets[shard_idx][source_idx]
        prompt = SIMILAR_RECORD_PROMPT.format(
            asm_code=asm,
            similar_asm_code=preprocessing_assembly(source["asm"]["code"][-1], remove_comments=True),
            similar_llvm_ir=preprocessing_llvm_ir(source["llvm_ir"]["code"][-1]),
        )
        (prompts_dir / f"sample_{idx}.txt").write_text(prompt)
        (retrieval_dir / f"sample_{idx}.json").write_text(json.dumps({
            "score": hit.score,
            "collection": args.collection_template.format(idx=shard_idx),
            "point_id": hit.id,
            "dataset_index": source_idx,
            "path": hit.payload.get("path", ""),
        }, indent=2) + "\n")
        jobs.append((idx, prompt))

    llm = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800)
    success = 0
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(generate_one, llm, args.model, prompt, args.max_tokens): idx
            for idx, prompt in jobs
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Generating LLVM IR", unit="sample"):
            idx = future_to_idx[future]
            try:
                llvm_ir, raw = future.result()
                (predictions_dir / f"sample_{idx}.ll").write_text(llvm_ir)
                (retrieval_dir / f"sample_{idx}.response.txt").write_text(raw)
                success += 1
            except Exception as exc:
                failures[idx] = str(exc)
                (retrieval_dir / f"sample_{idx}.generation.error").write_text(str(exc))

    summary = f"retrieved: {len(jobs)}\ngenerated: {success}\ngeneration_failures: {len(failures)}\nfailed_indices: {sorted(failures)}\n"
    (output_dir / "generation_summary.txt").write_text(summary)
    print(summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
