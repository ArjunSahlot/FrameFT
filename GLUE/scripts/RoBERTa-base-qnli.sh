#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset qnli \
  --task qnli \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 30 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name qnli-base \
  --output_dir /nas/frameFT/glue/qnli \
  --head_lr 0.00486 \
  --fft_lr 0.05 \
  --scale 70 \
  --tff_l 2 \
  --tff_block_size 4
