#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

gpu=$1
shift

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/nas/frameFT/spp/hf_datasets_cache}

tasks=arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,rte,winogrande
bs=${EVAL_BS:-4}
out_root=${EVAL_OUT:-/nas/frameFT/spp/lm_eval}

mkdir -p "${out_root}/logs"

for ckpt in "$@"; do
    p=${ckpt%/}
    if [ "$(basename "$p")" = "merged_model" ]; then
        name=$(basename "$(dirname "$p")")
    else
        name=$(basename "$p")
    fi

    log=${out_root}/logs/${name}.log

    if [ -f "${out_root}/${name}/.done" ]; then
        echo "=== SKIP ${name} already evaluated ==="
        continue
    fi

    if [ ! -f "${p}/config.json" ]; then
        echo "=== MISSING ${name} no model at ${p} ==="
        continue
    fi

    echo "=== START ${name} gpu=${gpu} $(date -u +%FT%TZ) ==="

    CUDA_VISIBLE_DEVICES=${gpu} python -m lm_eval \
        --model hf \
        --model_args pretrained=${p},trust_remote_code=True \
        --tasks ${tasks} \
        --device cuda:0 \
        --batch_size ${bs} \
        --output_path ${out_root}/${name} \
        > "${log}" 2>&1
    rc=$?

    if [ ${rc} -ne 0 ]; then
        echo "=== EVAL FAILED ${name} rc=${rc} $(date -u +%FT%TZ) see ${log} ==="
        continue
    fi

    avg=$(grep -A1 "^avg accs$" "${log}" | tail -1)
    accs=$(grep -B1 "^avg accs$" "${log}" | head -1 | tr -d '[],')

    if [ -z "${avg}" ]; then
        echo "=== NO AVG ${name} $(date -u +%FT%TZ) ==="
        continue
    fi

    touch "${out_root}/${name}/.done"
    echo "${name},${avg}" >> "${out_root}/summary.csv"
    echo "=== DONE ${name} avg=${avg} $(date -u +%FT%TZ) ==="
    echo "=== ACCS ${name} ${accs} ==="
done

echo "=== EVAL QUEUE COMPLETE gpu=${gpu} $(date -u +%FT%TZ) ==="
