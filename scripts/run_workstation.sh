#!/usr/bin/env bash
# 在一张 24-32 GB 的本地显卡上 (RTX 3090/4090/5090 等) 一条命令跑通任一实验。
# 在 Linux 或 Windows 的 WSL2 里执行 (Windows 原生 bash 也可, 但建议 WSL2):
#   bash scripts/run_workstation.sh setup            # 第一次: 建 venv、装 torch + LLaMA-Factory、下数据
#   bash scripts/run_workstation.sh baseline         # 实验 0: 基座三张表
#   bash scripts/run_workstation.sh A                # 实验 A: SLAKE SFT + 评估
#   bash scripts/run_workstation.sh B1               # 文本 CPT -> SFT + 评估
#   bash scripts/run_workstation.sh B2 /path/to/iu   # 图文 CPT (IU X-Ray 目录) -> SFT + 评估
# 结果: outputs/eval/*_<tag>.json, 拷回 results/ 即可对比。
set -euo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?setup|baseline|A|B1|B2}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$STEP" = setup ]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  # RTX 50 系列 (Blackwell) 需要 cu128 的 torch; 30/40 系列用 cu124 也可
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt hf_transfer
  pip install "llamafactory[torch,metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git"
  python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, 'GB')"
  python data/download_slake.py && python data/convert_slake_sharegpt.py
  python data/download_pubmedqa.py && python data/convert_pubmedqa_cpt.py --max 10000
  python scripts/make_bf16_configs.py
  echo "== setup 完成"; exit 0
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/make_bf16_configs.py >/dev/null
C=configs/bf16
clean_ckpt() { rm -rf "$1"/checkpoint-*; }

case "$STEP" in
  baseline)
    bash train/eval_all.sh baseline ;;
  A)
    llamafactory-cli train $C/sft_slake_qlora.yaml; clean_ckpt outputs/sft_slake_qlora_r16
    bash train/eval_all.sh sft_r16 outputs/sft_slake_qlora_r16 ;;
  B1)
    llamafactory-cli train $C/cpt_pubmed_qlora.yaml; clean_ckpt outputs/cpt_pubmed_qlora_r16
    llamafactory-cli train $C/sft_after_cpt.yaml;    clean_ckpt outputs/sft_after_cpt_r16
    bash train/eval_all.sh cpt_sft_r16 outputs/sft_after_cpt_r16 ;;
  B2)
    IU="${2:?IU X-Ray 目录 (含 indiana_reports.csv)}"
    python data/convert_iu_xray.py --root "$IU" --max 5000
    llamafactory-cli train $C/cpt_iu_qlora.yaml;     clean_ckpt outputs/cpt_iu_qlora_r16
    llamafactory-cli train $C/sft_after_cpt_iu.yaml; clean_ckpt outputs/sft_after_cpt_iu_r16
    bash train/eval_all.sh cpt_iu_sft_r16 outputs/sft_after_cpt_iu_r16 ;;
  *) echo "未知步骤 $STEP"; exit 1 ;;
esac
echo "== 完成: outputs/eval/ 下的 json 拷回 results/ 即可"
