#!/usr/bin/env bash
# 对一个 adapter (或基座) 跑完整四张评估表 (MMBench 是 2026-09-22 加的第二个通用探针)。用法:
#   bash train/eval_all.sh baseline                 # 基座
#   bash train/eval_all.sh sft_r16 outputs/sft_slake_qlora_r16
# 远程接口 (模型部署在别处, 见 docs/API.md):
#   MEDVLM_API_BASE=http://127.0.0.1:8000/v1 MODEL=<服务端模型名> bash train/eval_all.sh big35b_base
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:?tag}"
ADAPTER="${2:-}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
A=(); [ -n "$ADAPTER" ] && A=(--adapter "$ADAPTER")

python eval/eval_slake.py    --model "$MODEL" --tag "$TAG" "${A[@]}"   # 医学能力
python eval/eval_general.py  --model "$MODEL" --tag "$TAG" "${A[@]}"   # 通用能力保持
python eval/eval_pubmedqa.py --model "$MODEL" --tag "$TAG" "${A[@]}"   # 文本推理/可靠性
python eval/eval_mmbench.py  --model "$MODEL" --tag "$TAG" "${A[@]}"   # 通用能力保持 (第二探针)
echo "== 结果在 outputs/eval/*_$TAG.json"
