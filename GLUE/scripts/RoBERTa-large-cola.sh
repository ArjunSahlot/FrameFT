#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

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
  --head_lr 0.01 \
  --fft_lr 0.864 \
  --scale 10.0 \
  --tff_l 4 \
  --tff_block_size 4
