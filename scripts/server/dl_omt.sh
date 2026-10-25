#!/usr/bin/env bash
# OpenMedText 29 本教材: 逐个文件从 hf-mirror 直接下载 (hf download 会去列清单, 分页链接指回 huggingface.co, 服务器连不上)
D=/home/ubuntu/chunqian/data/cpt_corpora/openmedtext
ok=0
while IFS= read -r f; do
  f=${f%$''}; [ -z "$f" ] && continue
  mkdir -p "$D/$(dirname "$f")"
  url="https://hf-mirror.com/datasets/ywchoi/OpenMedText/resolve/main/$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$f")"
  for try in 1 2 3 4 5; do
    curl -sSL --fail -A "huggingface_hub/1.33.0" --connect-timeout 30 -o "$D/$f" "$url" && { ok=$((ok+1)); break; }
    sleep $((try * 20))
  done
  sleep 15
done < "${1:-/home/ubuntu/chunqian/scripts/omt_files.txt}"
echo "downloaded $ok"; find "$D/Med-Textbooks" -name '*.txt' | wc -l; du -sh "$D/Med-Textbooks"
