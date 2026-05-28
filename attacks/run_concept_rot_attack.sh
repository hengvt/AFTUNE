#!/usr/bin/env bash
set -euo pipefail

ATTACKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFTUNE_ROOT="$(cd "${ATTACKS_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 "${ATTACKS_DIR}/concept_rot_attack_finetuned.py" \
  -p "${AFTUNE_ROOT}/finetuned_model" \
  -o "${ATTACKS_DIR}/concept_rot_edited_model" \
  --device cuda:0 \
  --concept "computer science" \
  --n_train 50 \
  --reduced_test_set
