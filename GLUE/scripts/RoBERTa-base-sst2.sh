#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

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
  --head_lr 0.00829 \
  --fft_lr 0.661 \
  --scale 10.0 \
  --tff_l 4 \
  --tff_block_size 768
