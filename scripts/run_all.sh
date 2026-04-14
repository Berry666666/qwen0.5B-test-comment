#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python3 src/prepare_data.py \
  --dataset_repo dirtycomputer/weibo_senti_100k \
  --output_dir data/processed \
  --seed 42

python3 src/train.py --config configs/train_qwen2_0_5b.json
python3 src/evaluate.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --data_dir data/processed \
  --max_length 256
