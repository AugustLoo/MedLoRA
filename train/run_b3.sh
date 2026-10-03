#!/usr/bin/env bash
# 实验 B3 一键跑 (user0 容器, 3B, 单卡): 每组 CheXpert Plus 图文 CPT → SLAKE SFT → 四张表评测; 两组只差 CPT 数据怎么挑。
#   top  : 对齐分数最高的 5,000 对      rand : 同一候选池里随机的 5,000 对
# 前置 (一次): scripts/build_b3_pool.py sample + convert、scripts/score_chexpert_pairs.py 打分、data/convert_chexpert_b3.py
# 用法 (在 tmux 里, 先 nvidia-smi 确认卡空闲):
#   bash train/run_b3.sh              # 两组依次跑
#   bash train/run_b3.sh top          # 只跑一组
#   GPU=1 bash train/run_b3.sh rand   # 换卡 (共用账号, 多卡并行先和同账号的人商量)
# 进度: tail -3 /workspace/chunqian/runs/b3.log; 训练步数见 outputs/<名字>/trainer_log.jsonl 最后一行
# 做完的步骤自动跳过 (看 adapter 与评测结果文件在不在), 中断后重跑同一条命令即可。
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian
export CUDA_VISIBLE_DEVICES="${GPU:-0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOG=/workspace/chunqian/runs/b3.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

for f in data/processed/b3_cpt_top.json data/processed/b3_cpt_rand.json; do
  [[ -f $f ]] || { echo "缺 $f, 先跑 python data/convert_chexpert_b3.py"; exit 1; }
done

ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(top rand)
for ARM in "${ARMS[@]}"; do
  [[ $ARM == top || $ARM == rand ]] || { echo "未知组 $ARM (top / rand)"; exit 1; }
  CPT=outputs/cpt_b3_${ARM}_r16
  SFT=outputs/sft_after_cpt_b3_${ARM}_r16
  TAG=b3_${ARM}
  log "==== B3-$ARM (GPU $CUDA_VISIBLE_DEVICES) ===="
  if [[ -f $CPT/adapter_model.safetensors ]]; then log "CPT 已有, 跳过"; else
    log "CPT 开始"
    llamafactory-cli train configs/bf16/cpt_b3_${ARM}.yaml >> /workspace/chunqian/runs/${TAG}_cpt.log 2>&1
    rm -rf $CPT/checkpoint-*
    log "CPT 完成: $(tail -1 $CPT/trainer_log.jsonl)"
  fi
  if [[ -f $SFT/adapter_model.safetensors ]]; then log "SFT 已有, 跳过"; else
    log "SFT 开始"
    llamafactory-cli train configs/bf16/sft_after_cpt_b3_${ARM}.yaml >> /workspace/chunqian/runs/${TAG}_sft.log 2>&1
    rm -rf $SFT/checkpoint-*
    log "SFT 完成: $(tail -1 $SFT/trainer_log.jsonl)"
  fi
  if [[ -f outputs/eval/mmbench_${TAG}.json ]]; then log "评测已有, 跳过"; else
    log "评测开始 (SLAKE / TextVQA / PubMedQA / MMBench)"
    bash train/eval_all.sh $TAG $SFT >> /workspace/chunqian/runs/${TAG}_eval.log 2>&1
    log "评测完成: $(python -c "import json;s=json.load(open('outputs/eval/slake_$TAG.json'))['metrics'];print('SLAKE 封闭',s['closed_acc'],'开放EM',s.get('open_em'),'| TextVQA',json.load(open('outputs/eval/textvqa_$TAG.json'))['textvqa_acc'],'| MMBench',json.load(open('outputs/eval/mmbench_$TAG.json'))['mmbench_acc'])")"
  fi
done
log "全部完成: ${ARMS[*]}"
