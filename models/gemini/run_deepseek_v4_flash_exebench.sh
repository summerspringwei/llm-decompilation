#!/usr/bin/env bash
# Run the three 164-sample ExeBench subsets through the shared decompilation
# pipeline.  Each sample receives one completion per attempt and at most ten
# total LLM calls: one initial attempt plus nine correction attempts.

set -euo pipefail

: "${ARK_STREAM_API_KEY:?Set ARK_STREAM_API_KEY before running this script.}"

readonly model="deepseek-v4-flash-ga-260731"
readonly timestamp="$(date +%Y%m%d-%H%M%S)"
readonly root="${HOME}/Projects/validation/${model}/${timestamp}"
readonly embedding_url="${EMBEDDING_URL:-http://localhost:8125/embed/batch}"

mkdir -p "${root}"

for dataset in \
  sampled_dataset_with_loops_and_only_one_bb_164 \
  sampled_dataset_without_loops_164 \
  sampled_dataset_with_loops_164; do
  python -m models.gemini.gemini_decompilation \
    --dataset_name "${dataset}" \
    --model "${model}" \
    --num_generate 1 \
    --num_retry 9 \
    --num_processes 1 \
    --prompt-type in-context-learning \
    --qdrant_host localhost \
    --qdrant_port 6333 \
    --embedding_url "${embedding_url}" \
    --collection_name_with_idx 'train_synth_rich_io_filtered_{idx}_preprocessed_hermessim' \
    --output_dir "${root}/${dataset}" \
    2>&1 | tee "${root}/${dataset}.log"
done
