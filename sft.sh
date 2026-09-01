#!/usr/bin/env bash
set -euo pipefail

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export OMP_NUM_THREADS=8        # 消掉 libgomp 警告
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # 单卡；允许外部选择物理 GPU
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

BASE_MODEL="${BASE_MODEL:-./models/Qwen2.5-1.5B}"
SFT_OUTPUT="${SFT_OUTPUT:-./output/sft}"
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 1 \
            sft.py \
            --base_model "$BASE_MODEL" \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir "$SFT_OUTPUT/${category}_plus" \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False
done
