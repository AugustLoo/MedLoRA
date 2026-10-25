#!/usr/bin/env bash
# 35B 三步训练的第①步: 知识预训练 (纯文字: 英文教材 + 临床指南 + 中文医学百科 / 语料), 方案见 docs/PLAN_35B_CPT.md。
# 分阶段运行 (ubuntu 主机, 在 tmux 里), 每一步都可以重跑, 做完的会跳过:
#   bash train/interns2/run_cpt35.sh data      # 切段 + 抽样 + 留验证集 (不占卡; 语料先用 scripts/server/dl_cpt.sh 下到 /home/ubuntu/chunqian/data/cpt_corpora)
#   bash train/interns2/run_cpt35.sh smoke     # 3 步冒烟, 看显存和每步耗时 (要 0-3 号卡: 先请世龙停服务)
#   bash train/interns2/run_cpt35.sh sweep     # 学习率扫描: 5e-5 / 1e-4 / 2e-4 各短跑 SWEEP_STEPS 步, 比验证损失
#   LR=1e-4 bash train/interns2/run_cpt35.sh train   # 正式训练 1 轮 (LR 按扫描结果填); 断了重跑会接着练
#   bash train/interns2/run_cpt35.sh status    # 看进度和扫描结果
# 本脚本从不停世龙的服务: 他的服务 (23333) 开着或 0-3 号卡被占时, 要卡的阶段直接拒绝运行。
# 进度日志: tail -5 /home/ubuntu/chunqian/logs/cpt35.log
set -Eeuo pipefail

ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
DATA=$ROOT/data/processed
MODEL=/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
TAG=interns2_cpt1
OUT=outputs/$TAG
LOG=$ROOT/logs/cpt35.log
GPUS="${GPUS:-0,1,2,3}"
TRAIN_FILE=$DATA/cpt35_stage1.json
VAL_FILE=$DATA/cpt35_stage1_val.json
SWEEP_LRS=(${SWEEP_LRS:-5e-5 1e-4 2e-4})
SWEEP_STEPS=${SWEEP_STEPS:-120}
mkdir -p "$ROOT/logs"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

CONDA_BASE="$(conda info --base 2>/dev/null || true)"; [[ -n $CONDA_BASE ]] || CONDA_BASE=/home/ubuntu/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/hf

need_data() { for f in "$TRAIN_FILE" "$VAL_FILE"; do [[ -f $f ]] || { log "缺 $f, 先运行: bash $0 data"; exit 1; }; done; }
need_gpus() {
  if ss -ltn | grep -q ":23333 "; then log "世龙的服务还开着 (23333), 先请他停掉; 本脚本不会去停它"; exit 1; fi
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', *' '$2 > 1000')"
  [[ -z "$busy" ]] || { log "0-3 号卡还被占着: $busy"; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPUS"
}
train_cmd() {  # 与现有 35B 模型同一组 LoRA 设置; 预训练: 整段算损失, 1 轮, 3% 预热 (与 3B 的 B1 相同)
  python train/interns2/train_lora.py --mode pt --data "$TRAIN_FILE" --eval-data "$VAL_FILE" \
    --batch 4 --accum 4 --epochs 1 --warmup 0.03 --output "$@"
}

case "${1:-}" in
data)
  [[ -f $TRAIN_FILE ]] && { log "已有 $TRAIN_FILE, 跳过 (要重做先删掉)"; exit 0; }
  log "切段与抽样"
  python data/convert_cpt35_stage1.py --tokenizer "$MODEL" --out "$DATA" "${@:2}" 2>&1 | tee -a "$ROOT/logs/$TAG.data.log"
  log "统计: $DATA/cpt35_stage1_stats.json"
  ;;
smoke)
  need_data; need_gpus
  log "冒烟 3 步"
  train_cmd "outputs/${TAG}_smoke" --lr 1e-4 --smoke-steps 3 2>&1 | tee "$ROOT/logs/$TAG.smoke.log" | tail -40
  ;;
sweep)
  need_data; need_gpus
  for lr in "${SWEEP_LRS[@]}"; do
    o="outputs/${TAG}_sweep_lr$lr"
    if [[ -f $o/adapter_model.safetensors ]]; then log "扫描 lr=$lr 已做, 跳过"; continue; fi
    log "扫描 lr=$lr, $SWEEP_STEPS 步"
    # 短跑: 用前 SWEEP_STEPS×16 段训练 1 轮 (余弦退火到 0), 开始和结束各算一次验证损失
    train_cmd "$o" --lr "$lr" --limit $((SWEEP_STEPS * 16)) --eval-steps $((SWEEP_STEPS / 2)) --save-steps 100000 \
      >> "$ROOT/logs/$TAG.sweep.log" 2>&1
    log "lr=$lr 验证损失: $(grep '"eval"' "$o/train_log.jsonl" | python -c "import sys,json;print(' → '.join(str(json.loads(l)['eval_loss']) for l in sys.stdin))")"
  done
  ;;
train)
  need_data; need_gpus
  [[ -n ${LR:-} ]] || { log "先跑 sweep, 再用 LR=... 指定学习率"; exit 1; }
  if [[ -f $OUT/adapter_model.safetensors ]]; then log "已有 adapter, 跳过训练"; exit 0; fi
  RESUME=(); [[ -f $OUT/checkpoint/state.pt ]] && RESUME=(--resume)
  log "训练开始 lr=$LR ${RESUME[*]:-}"
  train_cmd "$OUT" --lr "$LR" --eval-steps 200 ${RESUME[@]+"${RESUME[@]}"} >> "$ROOT/logs/$TAG.train.log" 2>&1
  log "训练完成: $(grep -v '"eval"' "$OUT/train_log.jsonl" | tail -1)"
  ;;
status)
  for f in "$TRAIN_FILE" "$VAL_FILE"; do printf "%-24s %s\n" "$(basename "$f")" "$([[ -f $f ]] && echo 有 || echo 缺)"; done
  [[ -f $DATA/cpt35_stage1_stats.json ]] && python -c "import json;d=json.load(open('$DATA/cpt35_stage1_stats.json'));[print(f'  {k:16s}', v) for k,v in d.items()]"
  for lr in "${SWEEP_LRS[@]}"; do
    f="outputs/${TAG}_sweep_lr$lr/train_log.jsonl"
    [[ -f $f ]] && echo "扫描 lr=$lr: $(grep '"eval"' "$f" | python -c "import sys,json;print(' → '.join(str(json.loads(l)['eval_loss']) for l in sys.stdin))")"
  done
  [[ -f $OUT/train_log.jsonl ]] && echo "训练: $(grep -v '"eval"' "$OUT/train_log.jsonl" | tail -1)"
  [[ -f $OUT/adapter_model.safetensors ]] && echo "adapter: 有" || echo "adapter: 无"
  ;;
*)
  echo "用法: $0 data | smoke | sweep | train | status"; exit 1 ;;
esac
