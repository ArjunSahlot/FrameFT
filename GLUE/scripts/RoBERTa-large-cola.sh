#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.01
FFT_LR=0.864
SCALE=10.0
TFF_L=4
COEFF_BLOCK_SIZE=4

python NLU_GLUE.py \
  --model_name_or_path roberta-large \
  --dataset cola \
  --task cola \
  --n_ff_coeffs 1000 \
  --max_length 256 \
  --num_epochs 80 \
  --bs 128 \
  --seed 42 \
  --share_entry \
  --exp_name cola-large \
  --output_dir /nas/frameFT/glue/cola \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
