#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_LR=0.005667
FFT_LR=0.09
SCALE=30.0
TFF_L=2
COEFF_BLOCK_SIZE=768

python NLU_GLUE.py \
  --model_name_or_path roberta-large \
  --dataset qnli \
  --task qnli \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 30 \
  --bs 16 \
  --seed 42 \
  --share_entry \
  --exp_name qnli-large \
  --output_dir /nas/frameFT/glue/qnli \
  --head_lr "$HEAD_LR" \
  --fft_lr "$FFT_LR" \
  --scale "$SCALE" \
  --tff_l "$TFF_L" \
  --coeff_block_size "$COEFF_BLOCK_SIZE"
