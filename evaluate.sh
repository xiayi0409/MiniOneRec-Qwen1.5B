#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash evaluate.sh MODEL_PATH [GPU_ID]" >&2
  exit 2
fi

MODEL_PATH="$1"
GPU_ID="${2:-0}"
CATEGORY="Industrial_and_Scientific"
DATA_ROOT="data/Amazon"
RESULT_DIR="results/$(basename "$MODEL_PATH")"

TEST_FILE=$(find "$DATA_ROOT/test" -maxdepth 1 -name "${CATEGORY}*11.csv" -print -quit)
INFO_FILE=$(find "$DATA_ROOT/info" -maxdepth 1 -name "${CATEGORY}*.txt" -print -quit)

if [[ -z "$TEST_FILE" || -z "$INFO_FILE" ]]; then
  echo "Missing test/info data. See data/README.md." >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python -u evaluate.py \
  --base_model "$MODEL_PATH" \
  --info_file "$INFO_FILE" \
  --category "$CATEGORY" \
  --test_data_path "$TEST_FILE" \
  --result_json_data "$RESULT_DIR/final_result_${CATEGORY}.json" \
  --batch_size 8 \
  --num_beams 50 \
  --max_new_tokens 256 \
  --length_penalty 0.0

python calc.py \
  --path "$RESULT_DIR/final_result_${CATEGORY}.json" \
  --item_path "$INFO_FILE"
