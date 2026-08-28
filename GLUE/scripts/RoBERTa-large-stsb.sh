#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

python NLU_GLUE.py \
  --model_name_or_path roberta-large \
  --dataset stsb \
  --task stsb \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 30 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name stsb-large \
  --output_dir /nas/frameFT/glue/stsb \
  --head_lr 0.00446 \
  --fft_lr 0.22286 \
  --scale 50.0 \
  --tff_l 2 \
  --tff_block_size 4
