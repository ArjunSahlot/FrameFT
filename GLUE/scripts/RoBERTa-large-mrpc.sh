#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

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
  --head_lr 0.00685 \
  --fft_lr 0.119 \
  --scale 50.0 \
  --tff_l 4 \
  --tff_block_size 2
