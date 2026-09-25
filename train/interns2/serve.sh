#!/usr/bin/env bash
# 用同学推理环境里的 LMDeploy, 为「我们合并好的模型」起一个接口服务, 供评估脚本调用。
# 参数与同学的 scripts/start_server.sh 完全相同 (pytorch 后端, tp 4, reasoning-parser default),
# 只换模型路径、端口 (默认 23334, 不和他的 23333 撞) 和日志位置; 他的环境只读使用, 不改任何文件。
#
# 用法:
#   bash train/interns2/serve.sh start /home/ubuntu/chunqian/merged/interns2_mix_300
#   bash train/interns2/serve.sh stop
#   环境变量: GPUS (默认 0,1,2,3), PORT (默认 23334)
set -Eeuo pipefail

SVC=/home/ubuntu/Large-Model-Service-Interns2
ENV_DIR="$SVC/envs/inference"
CUDA_TOOLKIT="${CUDA_HOME:-/home/ubuntu/LSL/conda_envs/cuda128}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-23334}"
LOG_DIR=/home/ubuntu/chunqian/logs
PID_FILE="$LOG_DIR/serve_${PORT}.pid"
mkdir -p "$LOG_DIR"

case "${1:-}" in
start)
    MODEL_DIR="${2:?用法: serve.sh start <合并后的模型目录>}"
    [[ -f "$MODEL_DIR/config.json" ]] || { echo "ERROR: $MODEL_DIR 里没有 config.json"; exit 1; }
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "ERROR: 端口 $PORT 上已经有我们起的服务 (PID $(cat "$PID_FILE")), 先 stop"; exit 1
    fi
    if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: 端口 $PORT 被占用"; exit 1; fi
    export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT"
    export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
    export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_SHM_DISABLE=0
    export CUDA_VISIBLE_DEVICES="$GPUS"
    LOG_FILE="$LOG_DIR/serve_${PORT}.log"
    echo "模型 $MODEL_DIR | GPU $GPUS | 端口 $PORT | 日志 $LOG_FILE"
    setsid "$ENV_DIR/bin/lmdeploy" serve api_server "$MODEL_DIR" \
        --trust-remote-code --backend pytorch --tp 4 --server-port "$PORT" \
        --reasoning-parser default --tool-call-parser interns2-preview \
        >"$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    for i in $(seq 1 180); do
        if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "ERROR: 服务启动失败, 日志最后 60 行:"; tail -n 60 "$LOG_FILE"; rm -f "$PID_FILE"; exit 1
        fi
        if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            echo; echo "READY  http://127.0.0.1:${PORT}/v1   (容器里用 http://172.17.0.1:${PORT}/v1)"
            curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 300; echo
            exit 0
        fi
        printf "."; sleep 2
    done
    echo; echo "ERROR: 6 分钟内没起来, 看日志 $LOG_FILE"; exit 1
    ;;
stop)
    [[ -f "$PID_FILE" ]] || { echo "端口 $PORT 上没有我们起的服务"; exit 0; }
    PID="$(cat "$PID_FILE")"
    # 只停我们自己记录的这个进程组, 不碰别人的进程
    kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
    for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    rm -f "$PID_FILE"; echo "已停止 (PID $PID)"
    ;;
*)
    echo "用法: serve.sh start <模型目录> | stop"; exit 1 ;;
esac
