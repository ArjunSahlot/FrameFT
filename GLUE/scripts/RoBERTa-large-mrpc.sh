#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.00685
FFT_LR=0.119
SCALE=50.0
TFF_L=4
COEFF_BLOCK_SIZE=2

python NLU_GLUE.py \
  --model_name_or_path roberta-large \
  --dataset mrpc \
  --task mrpc \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 100 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name mrpc-large \
  --output_dir /nas/frameFT/glue/mrpc \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
