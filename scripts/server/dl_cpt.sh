#!/usr/bin/env bash
# 第①步知识预训练语料下载 (服务器连不上 huggingface.co, 走 hf-mirror.com); 断了重跑本脚本即可续传。
# 用 huggingface_hub 的 hf download (自带续传、校验与规范请求头; 直接 curl 镜像会被 403 / 429)。
# 放在 /home/ubuntu/chunqian/data/cpt_corpora/ 下:
#   openmedtext/Med-Textbooks/<许可>/<书名>.txt   29 本英文开源教材 (~34 MB)
#   guidelines/open_guidelines.jsonl               3.7 万篇临床指南 (~878 MB)
#   mmedc/Chinese.zip                              MMedC 中文部分 (~4.6 GB, 之后只解压教材)
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
HF=/home/ubuntu/chunqian/envs/s2train/bin/hf
D=/home/ubuntu/chunqian/data/cpt_corpora
mkdir -p "$D"
get() {  # get <repo> <输出目录> <文件或通配...>
  repo=$1 out=$2; shift 2
  for try in 1 2 3 4 5 6 7 8 9 10; do
    "$HF" download "$repo" --repo-type dataset --local-dir "$out" "$@" && return 0
    echo "  重试 $try ($repo)"; sleep $((try * 30))
  done
  echo "!! 失败: $repo"; return 1
}
echo "== $(date '+%F %T') OpenMedText 教材"
get ywchoi/OpenMedText "$D/openmedtext" --include "Med-Textbooks/*" --include README.md
echo "== $(date '+%F %T') Clinical Guidelines"
get epfl-llm/guidelines "$D/guidelines" open_guidelines.jsonl LICENSE README.md
echo "== $(date '+%F %T') MMedC 中文"
get Henrychur/MMedC "$D/mmedc" Chinese.zip README.md
echo "== $(date '+%F %T') 完成"
du -sh "$D"/*
find "$D/openmedtext" -name '*.txt' | wc -l
ls -la "$D"/guidelines "$D"/mmedc
