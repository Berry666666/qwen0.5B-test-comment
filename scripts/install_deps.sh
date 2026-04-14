#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user -r requirements.txt

echo "Dependencies installed with Tsinghua mirror."
