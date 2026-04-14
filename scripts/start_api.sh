#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
MODEL_DIR="${1:-outputs/qwen2.5-0.5b-weibo-senti/final}"
PORT="${2:-8000}"

python3 src/app.py --model_dir "$MODEL_DIR" --port "$PORT"
