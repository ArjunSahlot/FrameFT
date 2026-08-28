#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_DATASETS_TRUST_REMOTE_CODE=1

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
  --head_lr 0.006 \
  --fft_lr 0.321 \
  --scale 10.0 \
  --tff_l 2 \
  --tff_block_size 2
