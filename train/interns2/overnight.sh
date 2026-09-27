#!/usr/bin/env bash
# 一晚跑完 (ubuntu 主机, 0-3 号卡, 先停同学的服务):
#   1. 两个已训练模型 (回放 0 / 回放 300) 的开放式探针 —— 合并好的模型还在 /home/ubuntu/chunqian/merged/
#   2. 三个补种子的训练 + 四张表 + 开放式探针: 对照 (s43)、只挂注意力 (attn_s43)、只练 1 轮 (ep1_s43)
# 可以反复运行, 做完的会跳过。进度: tail -5 /home/ubuntu/chunqian/logs/ablation.log
# 前置: COCO 已经在主机上下载过一次 (白天跑基座开放式探针时会下), pubmedqa_sft_train_300s43.json 已拷到 data/processed。
set -Eeuo pipefail
ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
LOG=$ROOT/logs/ablation.log
cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
trap 'bash train/interns2/serve.sh stop >/dev/null 2>&1 || true' EXIT

[[ -f $ROOT/data/processed/pubmedqa_sft_train_300s43.json ]] || { log "缺 pubmedqa_sft_train_300s43.json, 先从容器拷过来"; exit 1; }

for T in interns2_mix_0 interns2_mix_300; do
  if [[ -f outputs/eval/openended_$T.json ]]; then log "开放式探针 $T: 已有, 跳过"; continue; fi
  [[ -f $ROOT/merged/$T/model.safetensors.index.json ]] || { log "缺合并模型 $ROOT/merged/$T, 跳过"; continue; }
  log "==== 开放式探针 $T ===="
  bash train/interns2/serve.sh start $ROOT/merged/$T >> $ROOT/logs/$T.serve.log 2>&1
  python eval/eval_openended.py --model $ROOT/merged/$T --tag $T >> $ROOT/logs/$T.openended.log 2>&1
  bash train/interns2/serve.sh stop >> $ROOT/logs/$T.serve.log 2>&1
  log "开放式探针完成 $T: $(python -c "import json;d=json.load(open('outputs/eval/openended_$T.json'));c,s=d['coco'],d['slake'];print('COCO CIDEr',c['cider_no_lp'],'词数',c['mean_words'],'短答%',c['short_lt5_pct'],'| SLAKE 部位%',s['location_ok_pct'],'异常说成正常',s['abnormal_called_normal'],'/',s['n_abnormal'])")"
done

OPENENDED=1 bash train/interns2/ablation.sh s43 attn_s43 ep1_s43
log "今晚的任务全部完成。记得: cd /home/ubuntu/Large-Model-Service-Interns2 && bash scripts/start_server.sh"
