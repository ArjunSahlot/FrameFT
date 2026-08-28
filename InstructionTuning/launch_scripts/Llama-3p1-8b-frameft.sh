#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python finetune.py \
    --base_model meta-llama/Llama-3.1-8B \
    --data_path data/alpaca_data_gpt4.json \
    --output_dir /nas/frameFT/spp/Frame/Llama-3.1-8B-frameft \
    --batch_size 128 \
    --cutoff_len 512 \
    --micro_batch_size 4 \
    --target_modules '[q_proj, v_proj]' \
    --num_epochs 3 \
    --wandb_project FrameFT \
    --n_ff_coeffs 5000 \
    --tff_l 2 \
    --tff_num_blocks 128 \
    --frame_scale 200.0 \
    --init_std 0.0 \
    --share_entry True \
    --entry_seed 2024 \
    --learning_rate 3e-2 \
    --wandb_run_name Llama-3.1-8B-frameft

python export_hf_checkpoint.py \
    --base_model meta-llama/Llama-3.1-8B \
    --ckpt_path /nas/frameFT/spp/Frame/Llama-3.1-8B-frameft

EVAL_BS=4 bash ../lm-evaluation-harness/launching_scripts/frame.sh 0 \
    /nas/frameFT/spp/Frame/Llama-3.1-8B-frameft/merged_model
