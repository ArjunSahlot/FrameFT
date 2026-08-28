#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export FRAMEFT_GRAD_CKPT=1
python finetune.py \
    --base_model meta-llama/Llama-2-13b-hf \
    --data_path data/alpaca_data_gpt4.json \
    --output_dir /nas/frameFT/spp/Frame/Llama-2-13b-hf-frameft \
    --batch_size 128 \
    --cutoff_len 512 \
    --micro_batch_size 4 \
    --target_modules '[q_proj, v_proj]' \
    --num_epochs 3 \
    --wandb_project FrameFT \
    --n_ff_coeffs 5000 \
    --tff_l 2 \
    --tff_block_size 40 \
    --tff_block_size_out 40 \
    --frame_scale 200.0 \
    --init_std 0.0 \
    --share_entry True \
    --entry_seed 2024 \
    --learning_rate 1e-1 \
    --wandb_run_name Llama-2-13b-hf-frameft

python export_hf_checkpoint.py \
    --base_model meta-llama/Llama-2-13b-hf \
    --ckpt_path /nas/frameFT/spp/Frame/Llama-2-13b-hf-frameft

EVAL_BS=2 bash ../lm-evaluation-harness/launching_scripts/frame.sh 0 \
    /nas/frameFT/spp/Frame/Llama-2-13b-hf-frameft/merged_model
