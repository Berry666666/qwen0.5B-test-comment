#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
LOGDIR="${1:-outputs/qwen2.5-0.5b-weibo-senti/runs}"
PORT="${2:-6006}"

~/.local/bin/tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
