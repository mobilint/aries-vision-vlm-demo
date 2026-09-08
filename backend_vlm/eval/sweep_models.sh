#!/bin/bash
# Evaluate the shipped prompt bundles across every available VLM model.
# Each VLM process serves one category (weapon on :5000, fall on :5001);
# run_eval derives the port from --category.
# Usage: ./sweep_models.sh [model ...]   (defaults to the full list)
set -euo pipefail
cd "$(dirname "$0")"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "CohereLabs/aya-vision-8b"
  )
fi

for model in "${MODELS[@]}"; do
  short=$(basename "$model" | tr '[:upper:]' '[:lower:]' | sed 's/-instruct$//')
  for category in weapon fall; do
    .venv/bin/python -u run_eval.py --category "$category" --model "$model" --label "sweep-$short"
  done
done

.venv/bin/python rescore.py
