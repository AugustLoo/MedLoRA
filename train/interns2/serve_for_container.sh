#!/usr/bin/env bash
# 给 user0 容器里的评测起 35B 服务 (CheXpert Plus 只在容器里, 不拷到主机; 容器经 172.17.0.1:23334 调用)。
# 一次一个模型: 合并 (仅 r0) → 起服务 → 打印容器里该跑的命令 → 等你评测完按回车 → 停服务、删临时合并模型。
#   base  : 基座                 /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
#   final : 最终版 (回放 300)    /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview-MedLoRA (只读使用)
#   r0    : 只做微调 (回放 0)    outputs/interns2_mix_0 临时合并到 /home/ubuntu/chunqian/merged/
#   r0_s43 / s43 : 回放 0 与回放 300 的第二个种子 (outputs/interns2_abl_r0_s43 / interns2_abl_s43), 同样临时合并
# 用法 (ubuntu 主机, tmux 里; 先停同学的服务):  bash train/interns2/serve_for_container.sh base
set -Eeuo pipefail
ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
MODELS=/home/ubuntu/Large-Model-Service-Interns2/models
cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" OMP_NUM_THREADS=1 HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TMP=""
case "${1:-}" in
  base)  TAG=interns2_base;    DIR=$MODELS/Intern-S2-Preview ;;
  final) TAG=interns2_mix_300; DIR=$MODELS/Intern-S2-Preview-MedLoRA ;;
  r0)    TAG=interns2_mix_0;   DIR=$ROOT/merged/interns2_mix_0; TMP=$DIR ;;
  r0_s43) TAG=interns2_abl_r0_s43; DIR=$ROOT/merged/$TAG; TMP=$DIR ;;
  s43)   TAG=interns2_abl_s43;  DIR=$ROOT/merged/$TAG; TMP=$DIR ;;
  six)   TAG=interns2_6ds;      DIR=$ROOT/merged/interns2_6ds ;;   # S2-6datasets, 合并模型保留, 不删
  *) echo "用法: $0 base | final | r0 | r0_s43 | s43 | six"; exit 1 ;;
esac
cleanup() { bash "$REPO/train/interns2/serve.sh" stop >/dev/null 2>&1 || true; [[ -n $TMP ]] && rm -rf "$TMP"; }
trap cleanup EXIT

if [[ -n $TMP && ! -f $TMP/model.safetensors.index.json ]]; then
  rm -rf "$TMP"
  echo "合并 outputs/$TAG (约 3 分钟) ..."
  python train/interns2/merge_lora.py --adapter outputs/$TAG --out "$TMP" > $ROOT/logs/$TAG.merge.log 2>&1
fi
bash train/interns2/serve.sh start "$DIR"
cat <<EOF

================ 服务已就绪: $1 ($TAG) ================
到 user0 容器里 (tmux), 运行:
  source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian && cd /workspace/chunqian/MedLoRA
  export MEDVLM_API_BASE=http://172.17.0.1:23334/v1 MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  python eval/eval_chexpert.py --model $DIR --tag $TAG --concurrency 8
容器里跑完 (打印出汇总) 后, 回到这里按回车: 停服务$( [[ -n $TMP ]] && echo "、删临时合并模型" )。
======================================================
EOF
read -r -p "容器里评测完成后按回车 ... " _
echo "已停服务: $TAG"
