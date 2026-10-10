#!/usr/bin/env bash
# S2-6datasets (interns2_6ds): 六个数据集的训练部分, 训练方法与 S2-2datasets (interns2_mix_300) 完全相同, 只换数据。
# 设计: docs/superpowers/specs/2026-10-08-s2-6datasets-design.md   计划: docs/superpowers/plans/2026-10-08-s2-6datasets.md
# 分阶段运行 (ubuntu 主机, 在 tmux 里), 每一步都可以重跑, 做完的会跳过:
#   bash train/interns2/run_6ds.sh data      # 下载四个数据集的训练部分并转换 (不占卡, 不用停世龙的服务)
#   bash train/interns2/run_6ds.sh smoke     # 3 步冒烟 (要 0-3 号卡: 先请世龙停服务)
#   bash train/interns2/run_6ds.sh train     # 正式训练, 约 6-8 小时; 断了重跑会接着练
#   bash train/interns2/run_6ds.sh merge     # 合并成完整模型 (约 73 GB, 保留, 给对比接口用)
#   bash train/interns2/run_6ds.sh eval      # 四张表: SLAKE / TextVQA / PubMedQA / MMBench
#   bash train/interns2/run_6ds.sh external  # VQA-RAD / PathVQA / MedQA / PneumoniaMNIST (对本模型是「领域内」)
#   bash train/interns2/run_6ds.sh status    # 看进度
# CheXpert 漏报评测在容器里跑: 主机 bash train/interns2/serve_for_container.sh six
# 本脚本从不停世龙的服务: 他的服务 (23333) 开着或 0-3 号卡被占时, 要卡的阶段直接拒绝运行。
# 进度日志: tail -5 /home/ubuntu/chunqian/logs/6ds.log
set -Eeuo pipefail

ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
DATA=$ROOT/data/processed
IMGS=$ROOT/data/sft6_images
PM=/workspace/chunqian/MedLoRA/data/raw/SLAKE=$ROOT/data/SLAKE
TAG=interns2_6ds
OUT=outputs/$TAG
MERGED=$ROOT/merged/$TAG
LOG=$ROOT/logs/6ds.log
GPUS="${GPUS:-0,1,2,3}"
FILES=(slake_train pubmedqa_sft_train_300 vqarad_sft_train pathvqa_sft_train medqa_sft_train pneumonia_sft_train)
mkdir -p "$ROOT/logs" "$ROOT/merged"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

CONDA_BASE="$(conda info --base 2>/dev/null || true)"; [[ -n $CONDA_BASE ]] || CONDA_BASE=/home/ubuntu/miniconda3  # 被别的脚本调用或非交互 shell 里没有 conda 函数
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/hf

DATA_FILES=()
for f in "${FILES[@]}"; do DATA_FILES+=("$DATA/$f.json"); done

need_data() {
  for f in "${DATA_FILES[@]}"; do [[ -f $f ]] || { log "缺 $f, 先运行: bash $0 data"; exit 1; }; done
}
need_gpus() {
  if ss -ltn | grep -q ":23333 "; then log "世龙的服务还开着 (23333), 先请他停掉; 本脚本不会去停它"; exit 1; fi
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', *' '$2 > 1000')"
  [[ -z "$busy" ]] || { log "0-3 号卡还被占着: $busy"; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPUS"
}
train_cmd() {  # 与 ablation.sh 主实验同一组参数, 只换 --data
  python train/interns2/train_lora.py --data "${DATA_FILES[@]}" --path-map "$PM" --batch 4 --accum 4 --output "$@"
}

case "${1:-}" in
data)
  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
  log "转换四个数据集的训练部分"
  python data/convert_sft6.py --out-dir "$DATA" --image-dir "$IMGS" "${@:2}" 2>&1 | tee -a "$ROOT/logs/$TAG.data.log"
  for f in "${DATA_FILES[@]}"; do
    [[ -f $f ]] && log "$(basename "$f"): $(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f") 条"
  done
  log "剔除与重叠统计: $DATA/sft6_stats.json"
  ;;
smoke)
  need_data; need_gpus
  log "冒烟 3 步"
  train_cmd "outputs/${TAG}_smoke" --smoke-steps 3 2>&1 | tee "$ROOT/logs/$TAG.smoke.log" | tail -40
  ;;
train)
  need_data; need_gpus
  if [[ -f $OUT/adapter_model.safetensors ]]; then log "已有 adapter, 跳过训练"; exit 0; fi
  RESUME=(); [[ -f $OUT/checkpoint/state.pt ]] && RESUME=(--resume)
  log "训练开始 ${RESUME[*]:-}"
  train_cmd "$OUT" ${RESUME[@]+"${RESUME[@]}"} >> "$ROOT/logs/$TAG.train.log" 2>&1
  log "训练完成: $(tail -1 "$OUT/train_log.jsonl")"
  ;;
merge)
  [[ -f $OUT/adapter_model.safetensors ]] || { log "还没有 adapter, 先 train"; exit 1; }
  if [[ -f $MERGED/model.safetensors.index.json ]]; then log "已合并, 跳过"; exit 0; fi
  FREE=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
  (( FREE >= 90 )) || { log "磁盘只剩 ${FREE} GB, 合并要约 75 GB, 先清理"; exit 1; }
  rm -rf "$MERGED"
  log "合并到 $MERGED"
  python train/interns2/merge_lora.py --adapter "$OUT" --out "$MERGED" >> "$ROOT/logs/$TAG.merge.log" 2>&1
  log "合并完成: $(grep -E '权重名|精度核对|完成' "$ROOT/logs/$TAG.merge.log" | tr '\n' ' ')"
  ;;
eval)
  [[ -f $MERGED/model.safetensors.index.json ]] || { log "还没有合并模型, 先 merge"; exit 1; }
  if [[ -f outputs/eval/mmbench_$TAG.json ]]; then log "四张表已有结果, 跳过"; exit 0; fi
  need_gpus
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
  export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  trap 'bash train/interns2/serve.sh stop >/dev/null 2>&1 || true' EXIT
  log "起服务"
  bash train/interns2/serve.sh start "$MERGED" >> "$ROOT/logs/$TAG.serve.log" 2>&1
  log "四张表开始"
  MODEL=$MERGED bash train/eval_all.sh $TAG >> "$ROOT/logs/$TAG.eval.log" 2>&1
  python scripts/eval_pubmedqa_split.py >> "$ROOT/logs/$TAG.eval.log" 2>&1 || true
  log "四张表完成: $(python -c "import json;print('SLAKE closed',json.load(open('outputs/eval/slake_$TAG.json'))['metrics']['closed_acc'],'| TextVQA',json.load(open('outputs/eval/textvqa_$TAG.json'))['textvqa_acc'],'| MMBench',json.load(open('outputs/eval/mmbench_$TAG.json'))['mmbench_acc'])")"
  ;;
external)
  [[ -f $MERGED/model.safetensors.index.json ]] || { log "还没有合并模型, 先 merge"; exit 1; }
  need_gpus
  bash train/interns2/external.sh six
  ;;
status)
  for f in "${DATA_FILES[@]}"; do printf "%-28s %s\n" "$(basename "$f")" "$([[ -f $f ]] && echo 有 || echo 缺)"; done
  [[ -f $OUT/train_log.jsonl ]] && echo "训练: $(tail -1 "$OUT/train_log.jsonl")"
  [[ -f $OUT/adapter_model.safetensors ]] && echo "adapter: 有" || echo "adapter: 无"
  [[ -f $MERGED/model.safetensors.index.json ]] && echo "合并模型: 有" || echo "合并模型: 无"
  ls outputs/eval/*_$TAG.json 2>/dev/null || echo "评测结果: 无"
  ;;
*)
  echo "用法: $0 data | smoke | train | merge | eval | external | status"; exit 1 ;;
esac
