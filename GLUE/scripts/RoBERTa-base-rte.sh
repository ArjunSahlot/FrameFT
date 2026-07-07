#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.006
FFT_LR=0.321
SCALE=10.0
TFF_L=2
COEFF_BLOCK_SIZE=2

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset rte \
  --task rte \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 60 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name rte-base \
  --output_dir /nas/frameFT/glue/rte \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
