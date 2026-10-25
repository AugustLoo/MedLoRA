#!/usr/bin/env bash
# 容器 (user0) 端, 与主机的 scripts/server/gpu_window.sh 配对: 等主机 23334 上出现指定的合并模型 → 跑 CheXpert Plus 漏报评测
# → 在内网放「完成」标记 (文件内容只有退出码, 不含任何数据)。由 s43_container.sh / eval35_container.sh 改成参数版。
# CheXpert Plus 是受控数据: 逐题记录 (chexpert_<tag>_preds.jsonl) 只留在容器里, 只有汇总 chexpert_<tag>.json 可以带回本机。
#
# 用法 (容器里; 共享账号, 一律用私有 tmux socket -L chunqian, 不要碰别人的 tmux, 不要用 pkill -f):
#   TAGS="interns2_s2a_s43 interns2_s2b_s43" END_AT="2026-11-01 14:10" \
#     tmux -L chunqian new-session -d -s cxpwatch "bash /workspace/chunqian/MedLoRA/scripts/server/container_cxp_watch.sh"
#   tail -3 /workspace/chunqian/MedLoRA/outputs/eval/cxp_watch.log    ← 第一行打印 CXP_SIG=..., 原样传给主机的 gpu_window.sh
# 环境变量:
#   TAGS     必填, 要测的模型标签 (= 主机 merged/ 下的目录名)
#   END_AT   必填, 北京时间; 到点不再等主机 (容器时钟是 UTC, 脚本内部换算)
#   SIG_TOKEN  可选, 标记地址里的随机串; 不填就新生成一个
#   BIND_IP    可选, 容器在内网的地址, 默认 172.20.0.2 (主机能访问到的那个)
X0=/workspace/chunqian/xfer3
BIND_IP=${BIND_IP:-172.20.0.2}
: "${TAGS:?要设 TAGS}"
: "${END_AT:?要设 END_AT (北京时间)}"
END=$(TZ=Asia/Shanghai date -d "$END_AT" +%s) || exit 1
SIG_TOKEN=${SIG_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_hex(16))')}
X=$X0/$SIG_TOKEN
mkdir -p "$X"; : > "$X0/index.html"   # 空的 index.html: 目录不能被列出来, 只有知道随机串才能取到标记
source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian && cd /workspace/chunqian/MedLoRA || exit 1
export MEDVLM_API_BASE=http://172.17.0.1:23334/v1 MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' PYTHONIOENCODING=utf-8
L=outputs/eval/cxp_watch.log
echo "CXP_SIG=http://$BIND_IP:38531/$SIG_TOKEN" | tee -a $L
if ss -ltn | grep -q ":38531 "; then
  echo "[$(date '+%m-%d %H:%M')] 38531 已有标记服务在跑 (它发布整个 xfer3 目录, 新随机串一样取得到), 沿用; 它到上次的 END_AT 后自动退出" | tee -a $L
else
  (cd $X0 && timeout $(( END - $(date +%s) + 600 )) python3 -m http.server 38531 --bind "$BIND_IP" > /dev/null 2>&1 &)
fi
echo "[$(date '+%m-%d %H:%M')] 开始等主机的服务: $TAGS" >> $L
while [ "$(date +%s)" -lt "$END" ]; do
  left=0; for t in $TAGS; do [ -f "$X/done_$t" ] || left=1; done
  [ $left -eq 0 ] && break
  M=$(curl -s -m 5 http://172.17.0.1:23334/v1/models | grep -o 'merged/interns2_[a-z0-9_]*' | head -1 | cut -d/ -f2)
  if [ -n "$M" ] && echo " $TAGS " | grep -q " $M " && [ ! -f "$X/done_$M" ]; then
    if [ -f "outputs/eval/chexpert_$M.json" ]; then echo 0 > "$X/done_$M"; continue; fi
    echo "[$(date '+%m-%d %H:%M')] CheXpert Plus 评测 $M" >> $L
    python eval/eval_chexpert.py --model "/home/ubuntu/chunqian/merged/$M" --tag "$M" --concurrency 8 >> $L 2>&1
    rc=$?; echo $rc > "$X/done_$M"
    echo "[$(date '+%m-%d %H:%M')] $M 结束, 退出码 $rc" >> $L
  fi
  sleep 30
done
echo "[$(date '+%m-%d %H:%M')] 结束 (CheXbert 复核另排, 见 AGENTS.md)" >> $L
