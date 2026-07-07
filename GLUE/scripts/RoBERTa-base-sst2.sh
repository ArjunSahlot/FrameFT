#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.00829
FFT_LR=0.661
SCALE=10.0
TFF_L=4
COEFF_BLOCK_SIZE=768

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset sst2 \
  --task sst2 \
  --n_ff_coeffs 1000 \
  --max_length 128 \
  --num_epochs 10 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name sst2-base \
  --output_dir /nas/frameFT/glue/sst2 \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
