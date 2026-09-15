#!/usr/bin/env bash
# 对一个 adapter (或基座) 跑完整三张评估表。用法:
#   bash train/eval_all.sh baseline                 # 基座
#   bash train/eval_all.sh sft_r16 outputs/sft_slake_qlora_r16
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:?tag}"
ADAPTER="${2:-}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
A=(); [ -n "$ADAPTER" ] && A=(--adapter "$ADAPTER")

python eval/eval_slake.py    --model "$MODEL" --tag "$TAG" "${A[@]}"   # 医学能力
python eval/eval_general.py  --model "$MODEL" --tag "$TAG" "${A[@]}"   # 通用能力保持
python eval/eval_pubmedqa.py --model "$MODEL" --tag "$TAG" "${A[@]}"   # 文本推理/可靠性
echo "== 结果在 outputs/eval/*_$TAG.json"
