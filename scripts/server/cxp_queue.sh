#!/usr/bin/env bash
# 主机端: 只为容器里的成人胸片评测 (CheXpert Plus) 轮流起服务。与容器里的 scripts/server/container_cxp_watch.sh 配对。
# 和 gpu_window.sh 的区别: 不停也不恢复世龙的服务, 只用 GPUS 指定的卡 (2026-11-23 起是 4-7 号), 主机这边不跑别的评测。
# 每个模型: 合并 (约 3 分钟) → 23334 起服务 → 等容器放「完成」标记 (约 50 分钟) → 停服务 → 删合并模型 (补丁保留)。
# 到 DEADLINE_AT 一定停下并清理, 保证卡空出来。容器 10 分钟没发请求就认为配对脚本没在跑, 全部停下, 不空占卡。
#
# 用法 (主机 tmux; 先在容器里起 container_cxp_watch.sh, 两边用同一个随机串):
#   GPUS=4,5,6,7 DEADLINE_AT='2026-11-24 05:30' CXP_SIG='http://172.20.0.2:38531/<随机串>' \
#     TAGS='interns2_abl_s44 interns2_abl_r0_s44' WAIT_TMUX=chunqian-w12 bash scripts/server/cxp_queue.sh
# 环境变量:
#   GPUS / DEADLINE_AT (北京时间) / CXP_SIG / TAGS  必填
#   WAIT_TMUX  可选, 先等这个 tmux 会话 (正在跑的训练) 结束再开始; 到截止时间它还没完就把它也停掉
# 进度: tail -5 /home/ubuntu/chunqian/logs/cxp_queue.log
set -Euo pipefail
ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
LOG=$ROOT/logs/cxp_queue.log
SERVE_LOG=$ROOT/logs/serve_23334.log
: "${GPUS:?要设 GPUS, 例如 GPUS=4,5,6,7}"
: "${DEADLINE_AT:?要设 DEADLINE_AT, 例如 DEADLINE_AT='2026-11-24 05:30'}"
: "${CXP_SIG:?要设 CXP_SIG (容器里 container_cxp_watch.sh 打印的地址)}"
: "${TAGS:?要设 TAGS}"
WAIT_TMUX=${WAIT_TMUX:-}
PER_MODEL=${PER_MODEL:-4200}     # 一个模型至少留 70 分钟, 不够就留到下次
NO_REQ=${NO_REQ:-600}            # 服务起来后这么久容器还没发请求 = 配对脚本没在跑
MAX_WAIT=${MAX_WAIT:-5400}       # 一个模型最多等 90 分钟
DEADLINE=$(date -d "$DEADLINE_AT" +%s) || exit 1
export GPUS PORT=23334
CUR=""; SERVING=0
mkdir -p "$ROOT/logs" "$ROOT/merged"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
marker() { curl -s -m 5 -f "$CXP_SIG/done_$1" 2>/dev/null | tr -dc 0-9; }
reqs() { local n; n=$(grep -c 'POST /v1/chat/completions' "$SERVE_LOG" 2>/dev/null); echo "${n:-0}"; }
busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | awk '$1 > 1000' | wc -l; }
cleanup() {
  if [[ $SERVING == 1 ]]; then bash "$REPO/train/interns2/serve.sh" stop >> "$LOG" 2>&1; SERVING=0; fi
  if [[ -n $CUR ]]; then rm -rf -- "$ROOT/merged/$CUR" && log "已删 $CUR 合并模型 (补丁保留)"; CUR=""; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "==== 胸片评测排队: $TAGS | 卡 $GPUS | 截止 $(date -d @"$DEADLINE" '+%m-%d %H:%M')${WAIT_TMUX:+ | 先等 tmux $WAIT_TMUX}"
if [[ -n $WAIT_TMUX ]]; then
  while tmux has-session -t "=$WAIT_TMUX" 2>/dev/null; do
    if (( $(left) <= 0 )); then
      log "!! 到截止时间 $WAIT_TMUX 还在跑, 停掉它 (训练每 100 步有断点, 下次接着训)"
      tmux kill-session -t "=$WAIT_TMUX" 2>/dev/null; sleep 60
      log "卡 $GPUS 上还占着显存的有 $(busy) 张"; exit 2
    fi
    sleep 60
  done
  log "$WAIT_TMUX 已结束"
fi
if ss -ltn | grep -q ":23334 "; then log "!! 23334 上还有服务, 不开始"; exit 1; fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"; [[ -n $CONDA_BASE ]] || CONDA_BASE=/home/ubuntu/miniconda3
set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ROOT/envs/s2train"; set -u
cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 HF_HOME=$ROOT/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for TAG in $TAGS; do
  rc=$(marker "$TAG"); if [[ -n $rc ]]; then log "$TAG: 容器里已有结果 (退出码 $rc), 跳过"; continue; fi
  [[ -f outputs/$TAG/adapter_model.safetensors ]] || { log "!! $TAG 还没有补丁, 跳过"; continue; }
  (( $(left) >= PER_MODEL )) || { log "离截止不足 $(( PER_MODEL / 60 )) 分钟, $TAG 及之后的留到下次"; break; }
  free_gb=$(df --output=avail -BG "$ROOT/merged/" | tail -1 | tr -dc 0-9)
  (( free_gb >= 80 )) || { log "!! 硬盘只剩 ${free_gb} GB, 合并要约 75 GB, 停下"; break; }
  DIR=$ROOT/merged/$TAG
  [[ ! -e $DIR ]] || { log "!! $DIR 已存在, 不覆盖也不清理, 停下"; break; }
  CUR=$TAG
  log "== $TAG: 合并"
  python train/interns2/merge_lora.py --adapter "outputs/$TAG" --out "$DIR" >> "$ROOT/logs/$TAG.merge.log" 2>&1 \
    || { log "!! 合并失败, 看 logs/$TAG.merge.log"; cleanup; continue; }
  SERVING=1
  bash train/interns2/serve.sh start "$DIR" >> "$LOG" 2>&1 || { log "!! 服务没起来"; cleanup; continue; }
  log "服务就绪, 等容器评测"
  t0=$(date +%s); stop_all=0
  while :; do
    rc=$(marker "$TAG"); if [[ -n $rc ]]; then log "$TAG: 容器评测结束, 退出码 $rc (共 $(reqs) 个请求, $(( ($(date +%s) - t0) / 60 )) 分钟)"; break; fi
    now=$(date +%s)
    if (( now - t0 > NO_REQ )) && (( $(reqs) == 0 )); then log "!! $(( NO_REQ / 60 )) 分钟内容器没发请求, 配对脚本可能没在跑, 全部停下"; stop_all=1; break; fi
    if (( now - t0 > MAX_WAIT )); then log "!! 等了 $(( MAX_WAIT / 60 )) 分钟容器还没完, 全部停下"; stop_all=1; break; fi
    if (( $(left) <= 0 )); then log "!! 到截止时间, 停下 $TAG 的评测"; stop_all=1; break; fi
    sleep 30
  done
  cleanup
  [[ $stop_all == 0 ]] || break
done
sleep 20
log "==== 结束。卡 $GPUS 上还占着显存的有 $(busy) 张"
