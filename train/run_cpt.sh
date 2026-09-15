#!/usr/bin/env bash
# 先 CPT 再 SFT 的完整两段式流程。用法: bash train/run_cpt.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f data/processed/pubmed_cpt.json ] || python data/convert_pubmedqa_cpt.py --max 20000
[ -f data/processed/slake_train.json ] || python data/convert_slake_sharegpt.py

echo "== 阶段 1: CPT"
llamafactory-cli train configs/cpt_pubmed_qlora.yaml
echo "== 阶段 2: SFT on top of CPT adapter"
llamafactory-cli train configs/sft_after_cpt.yaml
