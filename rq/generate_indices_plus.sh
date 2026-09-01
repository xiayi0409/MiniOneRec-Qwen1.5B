#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash generate_indices_plus.sh CHECKPOINT_PATH [DEVICE]" >&2
  exit 2
fi

CHECKPOINT_PATH="$1"
DEVICE="${2:-cuda:0}"

python generate_indices_plus.py \
  --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --ckpt_path "$CHECKPOINT_PATH" \
  --num_emb_list 256 256 256 \
  --device "$DEVICE"
