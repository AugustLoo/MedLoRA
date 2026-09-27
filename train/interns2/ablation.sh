#!/usr/bin/env bash
# D 组消融 (35B, 回放 300 设置上每次只改一样), 全自动: 训练 → 合并 → 起服务 → 四张表评估 → 停服务 → 删合并模型。
# 全部在 ubuntu 主机上完成, 0-3 号卡 (方案 A: 先停同学的服务)。可以反复运行: 已有 adapter 就跳过训练,
# 已有评估结果就跳过评估; 训练断了会带 --resume 从检查点接着跑。对照组是已经跑过的 interns2_mix_300, 不重跑。
#
# 用法 (在 tmux 里):
#   bash train/interns2/ablation.sh                # 全部 6 轮, 按下面 CONFIGS 的顺序
#   bash train/interns2/ablation.sh ep1 attn       # 只跑指定的几轮
#   OPENENDED=1 bash train/interns2/ablation.sh s43   # 评估时顺带跑开放式探针 (eval/eval_openended.py)
# 进度: tail -3 /home/ubuntu/chunqian/logs/ablation.log
# 前置 (一次): docs/INTERNS2.md「D 组消融」一节 —— 评估数据放到主机、s2train 环境补装 datasets / scikit-learn。
set -Eeuo pipefail

ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
DATA=$ROOT/data/processed
PM=/workspace/chunqian/MedLoRA/data/raw/SLAKE=$ROOT/data/SLAKE
LOG=$ROOT/logs/ablation.log
GPUS="${GPUS:-0,1,2,3}"
mkdir -p "$ROOT/logs" "$ROOT/merged"

# 名字 → 相对主实验 (rank 16 / alpha 32 / lr 1e-4 / 3 轮 / 注意力+共享专家) 改动的参数
declare -A CONFIGS=(
  [ep1]="--epochs 1"
  [attn]="--targets attn"
  [r8]="--rank 8 --alpha 16"
  [r32]="--rank 32 --alpha 64"
  [lr5e-5]="--lr 5e-5"
  [lr2e-4]="--lr 2e-4"
  # 第二个种子: 回放样本的抽样和训练种子一起换成 43 (与 3B 的 C2-300-s43 同一做法)
  [s43]="--seed 43 --data $DATA/slake_train.json $DATA/pubmedqa_sft_train_300s43.json"
  [attn_s43]="--targets attn --seed 43 --data $DATA/slake_train.json $DATA/pubmedqa_sft_train_300s43.json"
  [ep1_s43]="--epochs 1 --seed 43 --data $DATA/slake_train.json $DATA/pubmedqa_sft_train_300s43.json"
  # 回放 0 (只用 SLAKE) 的第二个种子: 检验「只做短答微调会把异常图说成正常」是否可重复
  [r0_s43]="--seed 43 --data $DATA/slake_train.json"
)
ORDER=(ep1 attn r8 r32 lr5e-5 lr2e-4)
[[ $# -gt 0 ]] && ORDER=("$@")

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cleanup() { bash "$REPO/train/interns2/serve.sh" stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

for NAME in "${ORDER[@]}"; do
  ARGS="${CONFIGS[$NAME]:?未知的消融名 $NAME}"
  TAG=interns2_abl_$NAME
  OUT=outputs/$TAG
  MERGED=$ROOT/merged/$TAG
  log "==== $NAME ($ARGS) ===="

  if [[ -f $OUT/adapter_model.safetensors ]]; then
    log "训练: 已有 adapter, 跳过"
  else
    RESUME=""; [[ -f $OUT/checkpoint/state.pt ]] && RESUME="--resume"
    log "训练开始 $RESUME"
    python train/interns2/train_lora.py --data $DATA/slake_train.json $DATA/pubmedqa_sft_train_300.json \
      --path-map "$PM" --batch 4 --accum 4 --output $OUT $ARGS $RESUME >> $ROOT/logs/$TAG.train.log 2>&1
    log "训练完成: $(tail -1 $OUT/train_log.jsonl)"
  fi

  NEED_STD=1; [[ -f outputs/eval/mmbench_$TAG.json ]] && NEED_STD=0
  NEED_OE=0; [[ "${OPENENDED:-0}" == 1 && ! -f outputs/eval/openended_$TAG.json ]] && NEED_OE=1
  if [[ $NEED_STD == 0 && $NEED_OE == 0 ]]; then
    log "评估: 已有结果, 跳过"; continue
  fi
  if [[ ! -f $MERGED/model.safetensors.index.json ]]; then
    rm -rf "$MERGED"
    log "合并"
    python train/interns2/merge_lora.py --adapter $OUT --out $MERGED >> $ROOT/logs/$TAG.merge.log 2>&1
    log "合并完成: $(grep -E '权重名|精度核对|完成' $ROOT/logs/$TAG.merge.log | tr '\n' ' ')"
  fi
  log "起服务"
  bash train/interns2/serve.sh start $MERGED >> $ROOT/logs/$TAG.serve.log 2>&1
  export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
  export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  if [[ $NEED_STD == 1 ]]; then
    log "评估开始"
    MODEL=$MERGED bash train/eval_all.sh $TAG >> $ROOT/logs/$TAG.eval.log 2>&1
    log "评估完成: $(python -c "import json;print('SLAKE closed',json.load(open('outputs/eval/slake_$TAG.json'))['metrics']['closed_acc'],'| PubMedQA F1 (1000 题口径)',json.load(open('outputs/eval/pubmedqa_$TAG.json'))['macro_f1'],'| TextVQA',json.load(open('outputs/eval/textvqa_$TAG.json'))['textvqa_acc'],'| MMBench',json.load(open('outputs/eval/mmbench_$TAG.json'))['mmbench_acc'])")"
  fi
  if [[ $NEED_OE == 1 ]]; then
    log "开放式探针开始"
    python eval/eval_openended.py --model $MERGED --tag $TAG >> $ROOT/logs/$TAG.openended.log 2>&1
    log "开放式探针完成: $(python -c "import json;d=json.load(open('outputs/eval/openended_$TAG.json'));c,s=d['coco'],d['slake'];print('COCO CIDEr',c['cider_no_lp'],'词数',c['mean_words'],'短答%',c['short_lt5_pct'],'| SLAKE 部位%',s['location_ok_pct'],'异常说成正常',s['abnormal_called_normal'],'/',s['n_abnormal'])")"
  fi
  bash train/interns2/serve.sh stop >> $ROOT/logs/$TAG.serve.log 2>&1
  rm -rf "$MERGED"
  log "已删除合并模型 (adapter 保留在 $OUT)"
done
log "全部完成: ${ORDER[*]}"
