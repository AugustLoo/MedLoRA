#!/usr/bin/env bash
# 在云端 (Kaggle / AutoDL / 实验室服务器) 跑 SFT。用法: bash train/run_sft.sh [config]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/sft_slake_qlora.yaml}"

# 数据没转换就先转
[ -f data/processed/slake_train.json ] || python data/convert_slake_sharegpt.py

echo "== 训练: $CFG"
llamafactory-cli train "$CFG"
