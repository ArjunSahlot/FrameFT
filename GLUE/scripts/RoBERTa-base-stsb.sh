#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.00657
FFT_LR=0.864
SCALE=10.0
TFF_L=4
COEFF_BLOCK_SIZE=2

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset stsb \
  --task stsb \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 30 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name stsb-base \
  --output_dir /nas/frameFT/glue/stsb \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
