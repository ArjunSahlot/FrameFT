#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset cola \
  --task cola \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 100 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name cola-base \
  --output_dir /nas/frameFT/glue/cola \
  --head_lr 0.00714 \
  --fft_lr 0.186 \
  --scale 30.0 \
  --tff_l 2 \
  --tff_block_size 768
