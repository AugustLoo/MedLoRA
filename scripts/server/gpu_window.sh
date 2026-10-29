#!/usr/bin/env bash
# 借用 0-3 号卡的一个「时间窗」: 停世龙的推理服务 → 依次训练 / 评测指定的 35B 模型 → 不管成功、失败还是被中断, 退出时恢复他的服务。
# 由实际跑过的接力脚本 (eval35_chain.sh 10-18、s43_chain.sh / s43_finish.sh 10-24) 合并而来: 流程不变, 只把截止时间和模型做成参数。
#
# 前提: 世龙已经同意停服务, 并约好还卡的时间。本脚本开头再看 2 分钟: 他的服务有新请求或卡在忙, 就不开始 (什么都不动)。
# 评测要先合并出约 75 GB 的完整模型, 合并目录 /home/ubuntu/chunqian/merged 所在的盘至少要有 80 GB (见 AGENTS.md「硬盘不够」)。
#
# 用法 (ubuntu 主机, 放进 tmux, 主机的 tmux 用默认 socket 即可):
#   tmux new-session -d -s gpuwin "DEADLINE_AT='2026-11-01 14:00' MODELS='a43 b43' \
#       bash /home/ubuntu/chunqian/MedLoRA/scripts/server/gpu_window.sh"
#   进度: tail -20 /home/ubuntu/chunqian/logs/gpu_window.log     训练进度: tail -1 MedLoRA/outputs/<TAG>/train_log.jsonl
# 环境变量:
#   DEADLINE_AT  必填, 还卡的时间 (主机时钟 = 北京时间)。到点立即停训练 / 评测 (训练每 100 步有断点, 之后可接着训);
#                离截止不足 1 小时不开始训练, 不足 2.2 小时不开始评测 (一个模型的评测约 2 小时)
#   MODELS       必填, train/interns2/run_sft35.sh 认识的模型, 按顺序做: a b c cpt1 a43 b43
#   STEPS        train,eval (默认) / train / eval / demo (人工试用, MODELS 只能选一个)
#                demo 不训练、不跑整套评测; 合并并开服务后等到截止时间, 退出时清理本次合并模型并恢复原服务
#   CXP_SIG      可选, 容器里 CheXpert Plus 评测的「完成标记」地址, 由 scripts/server/container_cxp_watch.sh 启动时打印;
#                不填就不等容器, 只做主机上的评测
set -u
ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
SVC=/home/ubuntu/Large-Model-Service-Interns2
LOG=$ROOT/logs/gpu_window.log
: "${DEADLINE_AT:?要设 DEADLINE_AT, 例如 DEADLINE_AT='2026-11-01 14:00'}"
: "${MODELS:?要设 MODELS, 例如 MODELS='a43 b43'}"
STEPS=${STEPS:-train,eval}
SIG=${CXP_SIG:-}
DEADLINE=$(date -d "$DEADLINE_AT" +%s) || exit 1
JOB=""
DEMO_DIR=""
mkdir -p "$ROOT/logs"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
want() { [[ ",$STEPS," == *",$1,"* ]]; }
tag_of() {  # 与 run_sft35.sh 的 model_of 一致
  case $1 in
    a) echo interns2_s2a ;; b) echo interns2_s2b ;; c) echo interns2_s3c ;; cpt1) echo interns2_cpt1 ;;
    a43) echo interns2_s2a_s43 ;; b43) echo interns2_s2b_s43 ;; *) return 1 ;;
  esac
}
wait_free() {
  for _ in $(seq 1 60); do
    [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1 > 1000' | wc -l) -eq 0 ]] && return 0
    sleep 10
  done
  return 1
}
stop_job() { [[ -n $JOB ]] && kill -0 "$JOB" 2>/dev/null && { kill -- -"$JOB" 2>/dev/null; sleep 20; kill -9 -- -"$JOB" 2>/dev/null; }; JOB=""; }
restore() {
  stop_job
  PORT=23334 bash "$REPO/train/interns2/serve.sh" stop >> "$LOG" 2>&1
  if [[ -n $DEMO_DIR ]]; then
    # 只清理 demo 本次新建的目录; 原本已有的合并模型在启动前就会被拒绝。
    rm -rf -- "$DEMO_DIR" && log "已删本次试用的合并模型 (补丁保留)"
  fi
  if ss -ltn | grep -q ":23333 "; then log "世龙的服务本来就在运行"; return; fi
  wait_free || log "!! 显存没释放干净, 仍尝试起服务"
  local s="svc23333_$(date +%m%d%H%M)"
  tmux new-session -d -s "$s" "bash $SVC/scripts/start_server.sh"
  for _ in $(seq 1 90); do
    if curl -s -m 5 http://127.0.0.1:23333/v1/models > /dev/null 2>&1; then log "世龙的服务已恢复 (tmux $s)"; return; fi
    sleep 10
  done
  log "!! 世龙的服务 15 分钟内没起来, 要人工看 (tmux $s)"
}
reqs() {  # 世龙的服务累计收到的请求数。grep -c 没匹配时输出 0 但返回 1, 不能写成 `|| echo 0` (会变成两行, 让下面的检查失效)
  local n; n=$(grep -c 'POST /v1/chat/completions' "$SVC/logs/inference.log" 2>/dev/null); echo "${n:-0}"
}
left() { echo $(( DEADLINE - $(date +%s) )); }
run_until_deadline() {  # run_until_deadline <命令...>: 新会话里跑, 过了截止时间就停; 返回 0 成功 / 1 失败 / 2 到点
  setsid "$@" >> "$LOG" 2>&1 &
  JOB=$!
  while kill -0 "$JOB" 2>/dev/null; do
    if (( $(left) <= 0 )); then log "!! 到截止时间, 停下: $*"; stop_job; return 2; fi
    sleep 30
  done
  wait "$JOB"; local rc=$?; JOB=""; return $rc
}
evaluate() {  # evaluate <模型> <TAG>: 合并 → 23334 起服务 → 主机评测 (和容器的 CheXpert Plus 同时) → 停服务 → 删合并模型
  local m=$1 tag=$2 dir=$ROOT/merged/$2
  (( $(left) >= 7920 )) || { log "离截止不足 2.2 小时, $tag 的评测留到下次"; return 2; }
  log "== $tag: 合并"
  bash train/interns2/run_sft35.sh merge "$m" >> "$LOG" 2>&1 || { log "!! 合并失败"; rm -rf "$dir"; return 1; }
  PORT=23334 bash train/interns2/serve.sh start "$dir" >> "$LOG" 2>&1 || { log "!! 服务没起来"; rm -rf "$dir"; return 1; }
  log "主机评测开始${SIG:+ (容器 CheXpert Plus 同时)}"
  run_until_deadline env KEEP_SERVE=1 bash train/interns2/run_sft35.sh eval "$m" &
  local hpid=$! host_rc="" cx_rc="" host_end=0 rc=0
  [[ -n $SIG ]] || cx_rc=-
  while :; do
    if [[ -z $host_rc ]] && ! kill -0 "$hpid" 2>/dev/null; then wait "$hpid"; host_rc=$?; host_end=$(date +%s); log "主机评测结束, 退出码 $host_rc"; fi
    if [[ -z $cx_rc ]]; then cx_rc=$(curl -s -m 5 -f "$SIG/done_$tag" 2>/dev/null | tr -dc 0-9); [[ -n $cx_rc ]] && log "容器 CheXpert Plus 结束, 退出码 $cx_rc"; fi
    [[ -n $host_rc && -n $cx_rc ]] && break
    if [[ -n $host_rc ]] && (( $(date +%s) - host_end > 5400 )); then log "!! 主机评测完 90 分钟容器还没完, 不等了"; break; fi
    if (( $(left) <= 0 )); then log "!! 到截止时间, 停下 $tag 的评测"; wait "$hpid"; rc=2; break; fi
    sleep 30
  done
  PORT=23334 bash train/interns2/serve.sh stop >> "$LOG" 2>&1
  wait_free || log "!! 显存没释放干净"
  rm -rf "$dir" && log "已删 $tag 合并模型 (补丁保留)"
  return $rc
}
demo() {  # demo <模型> <TAG>: 全程受截止时间限制, 不改训练补丁。
  local m=$1 tag=$2 dir=$ROOT/merged/$2 rc
  [[ ! -e $dir && ! -L $dir ]] || { log "!! $dir 已存在, 不覆盖也不清理, 请先核对"; return 1; }
  DEMO_DIR=$dir
  log "== $tag: 合并并限时试用"
  run_until_deadline bash train/interns2/run_sft35.sh merge "$m"
  rc=$?; [[ $rc -eq 0 ]] || { log "!! 合并未完成, 退出码 $rc"; return "$rc"; }
  run_until_deadline env PORT=23334 bash train/interns2/serve.sh start "$dir"
  rc=$?; [[ $rc -eq 0 ]] || { log "!! 服务未就绪, 退出码 $rc"; return "$rc"; }
  log "READY 人工试用: $tag, 端口 23334, 截止 $DEADLINE_AT; 可在独立终端提问"
  while (( $(left) > 0 )); do sleep 5; done
  log "到截止时间, 结束人工试用"
  return 2
}

case $STEPS in train|eval|train,eval|demo) ;; *) echo "STEPS 只能是 train / eval / train,eval / demo"; exit 1 ;; esac
for m in $MODELS; do tag_of "$m" > /dev/null || { echo "不认识的模型 $m (只能是 a b c cpt1 a43 b43)"; exit 1; }; done
cd "$REPO"
if want demo; then
  read -r -a demo_models <<< "$MODELS"
  [[ ${#demo_models[@]} -eq 1 ]] || { echo "demo 一次只能试用一个模型"; exit 1; }
  demo_tag=$(tag_of "${demo_models[0]}")
  [[ -f outputs/$demo_tag/adapter_model.safetensors && -f outputs/$demo_tag/adapter_config.json ]] || { echo "缺少补丁或 adapter_config.json"; exit 1; }
  demo_target=$ROOT/merged/$demo_tag
  [[ ! -e $demo_target && ! -L $demo_target ]] || { echo "$demo_target 已存在, 不覆盖也不清理"; exit 1; }
  ss -ltn | grep -q ":23334 " && { echo "23334 已有服务, 先核对, 不开始试用"; exit 1; }
  free_gb=$(df --output=avail -BG "$ROOT/merged/" | tail -1 | tr -dc 0-9)
  [[ $free_gb =~ ^[0-9]+$ ]] && (( free_gb >= 80 )) || { echo "合并目录所在盘至少要有 80 GB 可用空间"; exit 1; }
  (( $(left) >= 900 )) || { echo "demo 距截止时间至少要有 15 分钟"; exit 1; }
fi
log "==== 时间窗开始: 模型 $MODELS, 步骤 $STEPS, 截止 $(date -d @$DEADLINE '+%m-%d %H:%M')"
r0=$(reqs); sleep 120; r1=$(reqs)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3 | sort -n | tail -1)
if (( r1 > r0 || util > 10 )); then log "世龙的服务 2 分钟内有 $((r1 - r0)) 个请求、最高利用率 ${util}%, 有人在用, 不开始"; exit 1; fi
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "停世龙的服务 (已征得同意, 截止 $DEADLINE_AT)"
bash "$SVC/scripts/stop_server.sh" >> "$LOG" 2>&1
wait_free || { log "显存 10 分钟没释放, 放弃"; exit 1; }

for m in $MODELS; do
  TAG=$(tag_of "$m")
  if want train; then
    (( $(left) >= 3600 )) || { log "离截止不足 1 小时, $TAG 的训练留到下次"; break; }
    log "== $TAG: 训练"
    run_until_deadline bash train/interns2/run_sft35.sh train "$m"; rc=$?
    [[ $rc -eq 2 ]] && break
    [[ $rc -eq 0 && -f outputs/$TAG/adapter_model.safetensors ]] || { log "!! $TAG 训练失败"; break; }
    log "$TAG 训练完成: $(grep -v '"eval"' outputs/$TAG/train_log.jsonl | tail -1)"
  fi
  if want eval; then
    [[ -f outputs/$TAG/adapter_model.safetensors ]] || { log "!! $TAG 还没有补丁, 跳过评测"; continue; }
    evaluate "$m" "$TAG"
  fi
  if want demo; then
    demo "$m" "$TAG"
    demo_rc=$?
    log "==== 人工试用结束, 退出码 $demo_rc, 恢复原服务"
    [[ $demo_rc -eq 2 ]] && demo_rc=0  # 正常到点结束
    exit "$demo_rc"
  fi
  (( $(left) > 0 )) || break
done
log "==== 时间窗结束, 恢复世龙的服务"
