#!/usr/bin/env bash
# 对比用服务: 一次只开一个 35B (原版或训练版), 给做对比界面的同学在服务器上调用 (说明见 docs/COMPARE_API.md)。
# 两个模型都是完整的 35B, 各占 4 张卡; 能用的 0-3 号卡平时是世龙的服务在用, 所以同一时间只能开一个。
#
# 用法 (ubuntu 主机):
#   bash train/interns2/compare_serve.sh switch base     # 原版 Intern-S2-Preview          → 显示为 intern-s2-base
#   bash train/interns2/compare_serve.sh switch final    # 训练版 (交给世龙部署的那个)     → 显示为 intern-s2-medlora
#   bash train/interns2/compare_serve.sh status          # 现在开的是哪个
#   bash train/interns2/compare_serve.sh stop            # 用完关掉 (之后记得告诉世龙可以开回他的服务)
#
# 规矩: 本脚本只启停我们自己记录的进程 (serve.sh 的 PID 文件), 从不停世龙的服务;
#       他的服务 (端口 23333) 开着或 0-3 号卡被占时直接拒绝启动。两个模型目录都只读使用。
#       服务参数与评测时完全相同 (serve.sh), 不另设模型名: /v1/models 返回模型目录路径, 按目录名认出是哪个模型。
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS=/home/ubuntu/Large-Model-Service-Interns2/models
export PORT=23334 GPUS=0,1,2,3
SHILONG_PORT=23333
SERVE="$REPO/train/interns2/serve.sh"
URL="http://127.0.0.1:${PORT}/v1"

model_dir() {
    case "$1" in
        base)  echo "$MODELS/Intern-S2-Preview" ;;
        final) echo "$MODELS/Intern-S2-Preview-MedLoRA" ;;
        *) return 1 ;;
    esac
}
model_name() {
    case "$1" in
        base)  echo intern-s2-base ;;
        final) echo intern-s2-medlora ;;
    esac
}

current() {  # 当前在 23334 上回答的模型 (intern-s2-base / intern-s2-medlora); 没开或正在启动则为空
    local id
    id="$(curl -fsS --max-time 5 "$URL/models" 2>/dev/null | grep -o '"id": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/' || true)"
    case "${id%/}" in
        "") ;;
        */Intern-S2-Preview) echo intern-s2-base ;;
        */Intern-S2-Preview-MedLoRA) echo intern-s2-medlora ;;
        *) echo "$id" ;;
    esac
}

listening() { ss -ltn | grep -q ":$1 "; }

gpu_busy() {  # 0-3 号卡里显存占用超过 1 GB 的, 每行一张
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', *' '$2 > 1000 {print "  GPU " $1 ": " $2 " MiB"}'
}

wait_released() {  # 刚停掉的模型要过一会儿才把显存放掉 (tp 4 的后台进程逐个退出), 最多等 2 分钟
    local i
    for i in $(seq 1 60); do
        if [[ -z "$(gpu_busy)" ]]; then (( i > 1 )) && echo " 已释放"; return 0; fi
        (( i == 1 )) && printf "等显存释放"
        printf "."; sleep 2
    done
    echo
}

check_free() {
    if listening "$SHILONG_PORT"; then
        echo "世龙的服务还开着 (端口 $SHILONG_PORT), 本脚本不会去停它。"
        echo "先通知世龙, 等他停掉 (或经他同意后运行 bash /home/ubuntu/Large-Model-Service-Interns2/scripts/stop_server.sh), 再切换。"
        exit 1
    fi
    local busy
    busy="$(gpu_busy)"
    if [[ -n "$busy" ]]; then
        echo "0-3 号卡上的显存还被占着, 先弄清是谁的再切换 (不要直接杀进程):"
        echo "$busy"
        echo "查看占用的进程: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
        exit 1
    fi
}

case "${1:-}" in
switch)
    WHICH="${2:-}"
    DIR="$(model_dir "$WHICH")" || { echo "用法: $0 switch base | final"; exit 1; }
    NAME="$(model_name "$WHICH")"
    if [[ "$(current)" == "$NAME" ]]; then
        echo "现在开的已经是 $NAME, 不用切换。地址 $URL"; exit 0
    fi
    bash "$SERVE" stop                      # 只停我们自己开的 (没开就什么也不做)
    if listening "$PORT"; then echo "端口 $PORT 被别的程序占着, 先查清楚: ss -ltnp | grep $PORT"; exit 1; fi
    listening "$SHILONG_PORT" || wait_released
    check_free
    echo "开 $NAME (约 3-6 分钟, 期间接口连不上是正常的) ..."
    bash "$SERVE" start "$DIR"
    echo
    echo "现在开的是 $NAME   地址 $URL"
    echo "用完记得: bash $0 stop, 然后告诉世龙可以开回他的服务。"
    ;;
status)
    NOW="$(current)"
    if [[ -n "$NOW" ]]; then echo "对比服务: $NOW   地址 $URL"
    elif listening "$PORT"; then echo "对比服务: 端口 $PORT 有程序在监听, 但还没就绪 (可能正在启动)"
    else echo "对比服务: 没有开"; fi
    if listening "$SHILONG_PORT"; then echo "世龙的服务: 开着 (端口 $SHILONG_PORT)"; else echo "世龙的服务: 没开"; fi
    ;;
stop)
    bash "$SERVE" stop
    wait_released
    if [[ -n "$(gpu_busy)" ]]; then echo "注意: 0-3 号卡显存 2 分钟后仍未释放:"; gpu_busy; fi
    echo "已关。记得告诉世龙可以开回他的服务。"
    ;;
*)
    echo "用法: $0 switch base | switch final | status | stop"; exit 1 ;;
esac
