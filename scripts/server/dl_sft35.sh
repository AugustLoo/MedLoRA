#!/usr/bin/env bash
# 第②③步数据下载 (走 hf-mirror; 写明文件名, hf download 就不会去列清单)。断了重跑即可续传。
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_XET=1
HF=/home/ubuntu/chunqian/envs/s2train/bin/hf
D=/home/ubuntu/chunqian/data/sft35_raw
get() {  # get <repo> <目录名> <文件...>
  repo=$1 out=$D/$2; shift 2
  for try in 1 2 3 4 5 6; do
    "$HF" download "$repo" --repo-type dataset --local-dir "$out" "$@" && return 0
    echo "  重试 $try ($repo)"; sleep $((try * 30))
  done
  echo "!! 失败: $repo"
}
get FreedomIntelligence/HuatuoGPT-sft-data-v1 huatuo_sft HuatuoGPT_sft_data_v1.jsonl README.md
get Flmc/DISC-Med-SFT disc_med_sft DISC-Med-SFT_released.jsonl README.md
get fzkuji/CMExam cmexam train.json valid.json test.json README.md
get openlifescienceai/medmcqa medmcqa data/train-00000-of-00001.parquet data/validation-00000-of-00001.parquet data/test-00000-of-00001.parquet README.md
get FreedomIntelligence/medical-o1-reasoning-SFT medical_o1 medical_o1_sft.json medical_o1_sft_Chinese.json README.md
echo "== $(date '+%F %T') 完成"; du -sh $D/*
