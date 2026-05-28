#!/usr/bin/env bash
set -euo pipefail

ATTACKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFTUNE_ROOT="$(cd "${ATTACKS_DIR}/.." && pwd)"
WPA_DIR="${ATTACKS_DIR}/BackdoorLLM/attack/WPA"
MODEL_PATH="${AFTUNE_ROOT}/finetuned_model"

cd "$WPA_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export alg_name=BADEDIT
export model_name=Meta-Llama-3-8B
export hparams_fname=LLAMA3-8B.json
export ds_name=sst
export dir_name=sst
export target=Negative
export trigger=tq
export out_name=llama31-8b-finetuned-sst
export num_batch=5
export model_path="${MODEL_PATH}"
export save_model_dir="${ATTACKS_DIR}/wpa_edited_model"

python3 -m experiments.evaluate_backdoor \
  --alg_name "$alg_name" \
  --model_name "$model_name" \
  --model_path "$model_path" \
  --hparams_fname "$hparams_fname" \
  --ds_name "$ds_name" \
  --dir_name "$dir_name" \
  --trigger "$trigger" \
  --out_name "$out_name" \
  --num_batch "$num_batch" \
  --target "$target" \
  --few_shot \
  --conserve_memory \
  --save_model \
  --save_model_dir "$save_model_dir"
