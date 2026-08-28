#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

python NLU_GLUE.py \
  --model_name_or_path roberta-base \
  --dataset mrpc \
  --task mrpc \
  --n_ff_coeffs 1000 \
  --max_length 512 \
  --num_epochs 100 \
  --bs 32 \
  --seed 42 \
  --share_entry \
  --exp_name mrpc-base \
  --output_dir /nas/frameFT/glue/mrpc \
  --head_lr 0.01028 \
  --fft_lr 0.078 \
  --scale 50.0 \
  --tff_l 2 \
  --tff_block_size 768
