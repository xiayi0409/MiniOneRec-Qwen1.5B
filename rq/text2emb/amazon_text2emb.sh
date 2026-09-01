#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-1}"
EMBED_MODEL="${EMBED_MODEL:-../../models/Qwen2.5-1.5B}"

accelerate launch --num_processes "$NUM_PROCESSES" amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon/index \
    --plm_checkpoint "$EMBED_MODEL"
