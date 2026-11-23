#!/usr/bin/env bash
# 执行方案第 1–2 周的对照实验, 无人值守版 (主机): 等上一批训练结束 → 起原版模型, 生成 R2 的改写句和 R1 甲的描述
# → 做成训练文件 → 按顺序 ablation.sh (训练 + 合并 + 四张表 + 开放式探针)。
# 不停也不恢复世龙的服务, 只用 GPUS 指定的卡。到 DEADLINE_AT 一定停下并清理, 保证卡空出来
# (训练每 100 步有断点, 下次重跑同一命令接着训; 训练文件已生成就不再生成)。
#
# 用法 (主机 tmux):
#   GPUS=4,5,6,7 DEADLINE_AT='2026-11-24 05:30' MODELS='rw20 dself rw20_s43 dself_s43' WAIT_TMUX=chunqian-w12 \
#     bash scripts/server/controls_night.sh
# 进度: tail -5 /home/ubuntu/chunqian/logs/controls_night.log
set -Euo pipefail
ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
D=$ROOT/data/processed
BASE=/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
SLAKE_PM=/workspace/chunqian/MedLoRA/data/raw/SLAKE=$ROOT/data/SLAKE
LOG=$ROOT/logs/controls_night.log
: "${GPUS:?要设 GPUS, 例如 GPUS=4,5,6,7}"
: "${DEADLINE_AT:?要设 DEADLINE_AT, 例如 DEADLINE_AT='2026-11-24 05:30'}"
: "${MODELS:?要设 MODELS, 例如 MODELS='rw20 dself'}"
WAIT_TMUX=${WAIT_TMUX:-}
MIN_LEFT=${MIN_LEFT:-1200}       # 离截止不到 20 分钟就不再开新的一轮
DEADLINE=$(date -d "$DEADLINE_AT" +%s) || exit 1
export GPUS PORT=23334
JOB=""
mkdir -p "$ROOT/logs" "$ROOT/merged"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | awk '$1 > 1000' | wc -l; }
stop_job() { [[ -n $JOB ]] && kill -0 "$JOB" 2>/dev/null && { kill -- -"$JOB" 2>/dev/null; sleep 20; kill -9 -- -"$JOB" 2>/dev/null; }; JOB=""; }
cleanup() {
  stop_job
  bash "$REPO/train/interns2/serve.sh" stop >> "$LOG" 2>&1
  for m in $MODELS; do [[ -d $ROOT/merged/interns2_abl_$m ]] && rm -rf -- "$ROOT/merged/interns2_abl_$m" && log "已删 interns2_abl_$m 的合并模型 (补丁保留)"; done
  sleep 15; log "==== 结束。卡 $GPUS 上还占着显存的有 $(busy) 张"
}
run_until_deadline() {  # 新会话里跑, 过了截止时间就停; 返回 0 成功 / 2 到点 / 其他 = 失败
  setsid "$@" >> "$LOG" 2>&1 &
  JOB=$!
  while kill -0 "$JOB" 2>/dev/null; do
    if (( $(left) <= 0 )); then log "!! 到截止时间, 停下: $*"; stop_job; return 2; fi
    sleep 30
  done
  wait "$JOB"; local rc=$?; JOB=""; return $rc
}

log "==== 对照实验: $MODELS | 卡 $GPUS | 截止 $(date -d @"$DEADLINE" '+%m-%d %H:%M')${WAIT_TMUX:+ | 先等 tmux $WAIT_TMUX}"
if [[ -n $WAIT_TMUX ]]; then
  while tmux has-session -t "=$WAIT_TMUX" 2>/dev/null; do
    if (( $(left) <= 0 )); then
      log "!! 到截止时间 $WAIT_TMUX 还在跑, 停掉它 (训练有断点, 下次接着训)"
      tmux kill-session -t "=$WAIT_TMUX" 2>/dev/null; sleep 60
      log "卡 $GPUS 上还占着显存的有 $(busy) 张"; exit 2
    fi
    sleep 60
  done
  log "$WAIT_TMUX 已结束"
fi
if ss -ltn | grep -q ":23334 "; then log "!! 23334 上还有服务, 不开始"; exit 1; fi
for _ in $(seq 1 30); do [[ $(busy) -eq 0 ]] && break; sleep 10; done
[[ $(busy) -eq 0 ]] || { log "!! 卡 $GPUS 上有别的程序占着显存, 不开始"; exit 1; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CONDA_BASE="$(conda info --base 2>/dev/null || true)"; [[ -n $CONDA_BASE ]] || CONDA_BASE=/home/ubuntu/miniconda3
set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ROOT/envs/s2train"; set -u
cd "$REPO" || exit 1
export OMP_NUM_THREADS=1 HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8

# ---- 1. 生成 (两个训练文件都已有就跳过) ----
if [[ ! -f $D/slake_train_rw20.json || ! -f $D/slake_desc_self.json ]]; then
  (( $(left) >= 1800 )) || { log "离截止不足 30 分钟, 生成留到下次"; exit 0; }
  log "起原版模型, 生成改写句和描述"
  bash train/interns2/serve.sh start "$BASE" >> "$LOG" 2>&1 || { log "!! 原版模型的服务没起来"; exit 1; }
  export MEDVLM_API_BASE=http://127.0.0.1:23334/v1 MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  run_until_deadline python scripts/gen_control_text.py rewrite "$D/rw_req.jsonl" "$D/rw_gen.jsonl" --model "$BASE"; rc1=$?
  run_until_deadline python scripts/gen_control_text.py describe "$D/desc_req.jsonl" "$D/desc_gen.jsonl" --model "$BASE" --path-map "$SLAKE_PM"; rc2=$?
  bash train/interns2/serve.sh stop >> "$LOG" 2>&1
  unset MEDVLM_API_BASE MEDVLM_API_EXTRA
  log "生成结束: 改写退出码 $rc1 ($(wc -l < "$D/rw_gen.jsonl" 2>/dev/null || echo 0) / $(wc -l < "$D/rw_req.jsonl") 条), 描述退出码 $rc2 ($(wc -l < "$D/desc_gen.jsonl" 2>/dev/null || echo 0) / $(wc -l < "$D/desc_req.jsonl") 条)"
  if [[ $rc1 -eq 0 ]]; then
    python data/make_short_answer_controls.py rw-apply "$D/slake_train.json" "$D/slake_train_rw20.json" --gen "$D/rw_gen.jsonl" >> "$LOG" 2>&1 \
      || log "!! 改写没过检查 (见上一行), 不生成 slake_train_rw20.json, rw20 两轮今晚不训"
  fi
  if [[ $rc2 -eq 0 ]]; then
    python data/make_short_answer_controls.py desc-apply "$D/slake_train.json" "$D/slake_desc_self.json" --gen "$D/desc_gen.jsonl" >> "$LOG" 2>&1 \
      || log "!! 描述没做成训练文件"
  fi
  for _ in $(seq 1 30); do [[ $(busy) -eq 0 ]] && break; sleep 10; done
fi

# ---- 2. 训练 + 评估, 一轮一轮来 ----
for m in $MODELS; do
  case $m in
    rw20*)  [[ -f $D/slake_train_rw20.json ]] || { log "$m: 没有 slake_train_rw20.json, 跳过"; continue; } ;;
    dself*) [[ -f $D/slake_desc_self.json ]]  || { log "$m: 没有 slake_desc_self.json, 跳过"; continue; } ;;
  esac
  (( $(left) >= MIN_LEFT )) || { log "离截止不足 $(( MIN_LEFT / 60 )) 分钟, $m 及之后的留到下次"; break; }
  log "== $m"
  run_until_deadline env OPENENDED=1 bash train/interns2/ablation.sh "$m"; rc=$?
  [[ $rc -eq 2 ]] && break
  [[ $rc -eq 0 ]] || log "!! $m 退出码 $rc, 看 logs/ablation.log 和 logs/interns2_abl_$m.*.log"
done
