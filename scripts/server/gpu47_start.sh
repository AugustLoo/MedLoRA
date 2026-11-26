#!/usr/bin/env bash
# gpu47 的 4 卡容器端 (2026-11-27 起代替 user0; 两个容器的 /workspace 是同一个, rafi 的胸片子集和缩好的图都在):
# 找一个装了 pillow / numpy / tqdm 的 python (没有就装到 /workspace/chunqian/pylib, 只动我们自己的目录)
# → 在私有 tmux (-L chunqian, 会话 cxp47) 里起 container_cxp_watch.sh, 与主机的 scripts/server/cxp_queue.sh 配对。
# gpu47 只能密码登录, 由用户自己登; 本机把要用的变量写在最前面, 再用 ssh ... bash -s < 文件 喂进去。
# 需要的变量: TAGS (空格分隔) / END_AT (北京时间) / SIG_TOKEN (和主机 CXP_SIG 里的一样)
# 地址: 容器 172.23.0.2, 主机在这个网段是 172.23.0.1 (2026-11-27 实查; 主机 ping 得通容器)。
set -u
: "${TAGS:?要设 TAGS}" "${END_AT:?要设 END_AT}" "${SIG_TOKEN:?要设 SIG_TOKEN}"
W=/workspace/chunqian
LIB=$W/pylib
cd $W/MedLoRA || exit 1
PY=""
for c in /home/gpu47/envs/research/bin/python3 /opt/conda/bin/python3; do
  if PYTHONPATH=$LIB $c -c 'import PIL, numpy, tqdm' 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  c=/home/gpu47/envs/research/bin/python3
  echo "== 装 pillow numpy tqdm 到 $LIB"
  $c -m pip install -q --disable-pip-version-check --timeout 30 --target $LIB pillow numpy tqdm \
    || $c -m pip install -q --disable-pip-version-check --timeout 30 --target $LIB -i https://pypi.tuna.tsinghua.edu.cn/simple pillow numpy tqdm
  PYTHONPATH=$LIB $c -c 'import PIL, numpy, tqdm' 2>/dev/null && PY=$c
fi
[ -n "$PY" ] || { echo "!! 没有能用的 python 包, 没开始"; exit 1; }
echo "== python: $PY"
PYTHONPATH=$LIB $PY -c 'import PIL, numpy, tqdm; print("PIL", PIL.__version__, "numpy", numpy.__version__, "tqdm", tqdm.__version__)'
echo "== 胸片: $(find $W/data/chexpert_test_1024 -type f | wc -l) 张 (应为 3189)"
timeout 3 bash -c '</dev/tcp/172.23.0.1/23333' 2>/dev/null && echo "== 连得到主机" || echo "!! 连不到主机 172.23.0.1"
if tmux -L chunqian has-session -t cxp47 2>/dev/null; then
  echo "== cxp47 已经在跑, 不重复开"
else
  cat > $W/run_cxp47.sh <<EOF
export TAGS='$TAGS' END_AT='$END_AT' SIG_TOKEN=$SIG_TOKEN BIND_IP=172.23.0.2 HOST_IP=172.23.0.1 PY=$PY PYTHONPATH=$LIB
exec bash $W/MedLoRA/scripts/server/container_cxp_watch.sh
EOF
  tmux -L chunqian new-session -d -s cxp47 "bash $W/run_cxp47.sh"
fi
sleep 8
echo "== tmux"; tmux -L chunqian ls 2>&1
echo "== 日志"; tail -3 $W/MedLoRA/outputs/eval/cxp_watch.log
echo "== 标记服务: $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://172.23.0.2:38531/)  (200 = 正常)"
