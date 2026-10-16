#!/usr/bin/env bash
# 35B 三步训练的第②③步与评测 (方案 docs/PLAN_35B_CPT.md; 第①步见 run_cpt35.sh)。三个模型都在同一套 LoRA 设置上:
#   a = 模型 A  (interns2_s2a): 只做第②步, 从原版开始                     → 对照组
#   b = 模型 B  (interns2_s2b): 第①步补丁 (interns2_cpt1) 接着做第②步       → B 对比 A: 知识预训练有没有用
#   c = 模型 C  (interns2_s3c): 模型 B 的补丁接着做第③步                    → C 对比 B: 推理训练有没有用
# 分阶段运行 (ubuntu 主机, 在 tmux 里), 每一步都可以重跑, 做完的会跳过:
#   bash train/interns2/run_sft35.sh data2     # 第②步中文问答 / 对话: 清洗、去重、测试集隔离、抽样 (不占卡)
#   bash train/interns2/run_sft35.sh data3     # 第③步带解析的考题 (不占卡)
#   bash train/interns2/run_sft35.sh smoke     # 第②步数据 3 步冒烟 (多轮对话 + 图片混在一起), 看显存和每步耗时
#   bash train/interns2/run_sft35.sh train a|b|c
#   bash train/interns2/run_sft35.sh merge a|b|c
#   bash train/interns2/run_sft35.sh eval  a|b|c        # 四张表 + SLAKE 中文 + CMExam + 四个外部测试
#   bash train/interns2/run_sft35.sh evalzh base|final|six   # 已有模型补测两项中文评测, 作对照
#   bash train/interns2/run_sft35.sh status
# CheXpert Plus 成人胸片评测在容器里跑 (同 S2-6datasets 的做法), 本脚本不管。
# 本脚本从不停世龙的服务: 他的服务 (23333) 开着或 0-3 号卡被占时, 要卡的阶段直接拒绝运行。
# 进度日志: tail -5 /home/ubuntu/chunqian/logs/sft35.log
set -Eeuo pipefail

ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
DATA=$ROOT/data/processed
MODELS=/home/ubuntu/Large-Model-Service-Interns2/models
TOKENIZER=$MODELS/Intern-S2-Preview
PM=/workspace/chunqian/MedLoRA/data/raw/SLAKE=$ROOT/data/SLAKE
LOG=$ROOT/logs/sft35.log
GPUS="${GPUS:-0,1,2,3}"
FILES6=(slake_train pubmedqa_sft_train_300 vqarad_sft_train pathvqa_sft_train medqa_sft_train pneumonia_sft_train)
STAGE2=("${FILES6[@]/#/$DATA/}"); STAGE2=("${STAGE2[@]/%/.json}"); STAGE2+=("$DATA/sft35_zh_dialog.json")
STAGE3=("$DATA/sft35_reasoning.json")
CPT=outputs/interns2_cpt1
mkdir -p "$ROOT/logs" "$ROOT/merged"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

CONDA_BASE="$(conda info --base 2>/dev/null || true)"; [[ -n $CONDA_BASE ]] || CONDA_BASE=/home/ubuntu/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/hf
# 建测试集索引要读 MedQA / PubMedQA (datasets 缓存), 评测也一样: 走镜像只发元数据请求, 文件都已缓存
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

need_files() { for f in "$@"; do [[ -f $f ]] || { log "缺 $f"; exit 1; }; done; }
need_gpus() {
  if ss -ltn | grep -q ":23333 "; then log "世龙的服务还开着 (23333), 先请他停掉; 本脚本不会去停它"; exit 1; fi
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', *' '$2 > 1000')"
  [[ -z "$busy" ]] || { log "0-3 号卡还被占着: $busy"; exit 1; }
  export CUDA_VISIBLE_DEVICES="$GPUS"
}
model_of() {  # a|b|c → TAG / 数据 / 起点补丁 / 学习率 / 轮数
  case $1 in
    a) TAG=interns2_s2a; DATA_FILES=("${STAGE2[@]}"); INIT=""; LR=1e-4; EPOCHS=3 ;;
    b) TAG=interns2_s2b; DATA_FILES=("${STAGE2[@]}"); INIT=$CPT; LR=1e-4; EPOCHS=3 ;;
    c) TAG=interns2_s3c; DATA_FILES=("${STAGE3[@]}"); INIT=outputs/interns2_s2b; LR=5e-5; EPOCHS=1 ;;
    *) echo "模型只能是 a / b / c"; exit 1 ;;
  esac
  OUT=outputs/$TAG; MERGED=$ROOT/merged/$TAG
}
train_cmd() {  # 与现有 35B 模型同一组 LoRA 设置, 只换数据 / 起点 / 学习率 / 轮数
  local init=(); [[ -n $INIT ]] && init=(--init-adapter "$INIT")
  python train/interns2/train_lora.py --data "${DATA_FILES[@]}" --path-map "$PM" --batch 4 --accum 4 \
    --lr "$LR" --epochs "$EPOCHS" ${init[@]+"${init[@]}"} --output "$@"
}

run_evals() {  # run_evals <TAG> <模型目录> [只测中文]
  local tag=$1 dir=$2 zh_only=${3:-}
  export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
  export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  trap 'bash train/interns2/serve.sh stop >/dev/null 2>&1 || true' EXIT
  log "起服务 $dir"
  bash train/interns2/serve.sh start "$dir" >> "$ROOT/logs/$tag.serve.log" 2>&1
  local L=$ROOT/logs/$tag.eval.log
  if [[ -z $zh_only && ! -f outputs/eval/mmbench_$tag.json ]]; then
    log "$tag 四张表"
    MODEL=$dir bash train/eval_all.sh "$tag" >> "$L" 2>&1
    python scripts/eval_pubmedqa_split.py >> "$L" 2>&1 || true
  fi
  [[ -f outputs/eval/slake_zh_$tag.json ]] || { log "$tag SLAKE 中文"; python eval/eval_slake.py --model "$dir" --tag "$tag" --lang zh >> "$L" 2>&1; }
  [[ -f outputs/eval/cmexam_$tag.json ]] || { log "$tag CMExam"; python eval/eval_cmexam.py --model "$dir" --tag "$tag" >> "$L" 2>&1; }
  if [[ -z $zh_only ]]; then
    for DS in vqa-rad path-vqa; do
      [[ -f outputs/eval/${DS//-/}_$tag.json ]] || { log "$tag $DS"; python eval/eval_medvqa.py --dataset $DS --model "$dir" --tag "$tag" >> "$L" 2>&1; }
    done
    [[ -f outputs/eval/medqa_$tag.json ]] || { log "$tag MedQA"; python eval/eval_medqa.py --model "$dir" --tag "$tag" >> "$L" 2>&1; }
    [[ -f outputs/eval/pneumonia_$tag.json ]] || { log "$tag PneumoniaMNIST"; python eval/eval_pneumonia.py --model "$dir" --tag "$tag" --freetext >> "$L" 2>&1; }
  fi
  bash train/interns2/serve.sh stop >> "$ROOT/logs/$tag.serve.log" 2>&1
  log "$tag 评测完成: $(python -c "
import json,os
def g(f,k):
    p='outputs/eval/'+f
    return json.load(open(p)).get(k) if os.path.exists(p) else None
print('SLAKE中文', (g('slake_zh_$tag.json','metrics') or {}).get('closed_acc'), '| CMExam', g('cmexam_$tag.json','cmexam_acc'),
      '| MedQA', g('medqa_$tag.json','medqa_acc'))")"
}

case "${1:-}" in
data2)
  [[ -f $DATA/sft35_zh_dialog.json ]] && { log "已有 sft35_zh_dialog.json, 跳过 (要重做先删掉)"; exit 0; }
  log "第②步数据"
  python data/convert_sft35_stage2.py --tokenizer "$TOKENIZER" --out "$DATA" "${@:2}" 2>&1 | tee -a "$ROOT/logs/sft35.data2.log"
  ;;
data3)
  [[ -f $DATA/sft35_reasoning.json ]] && { log "已有 sft35_reasoning.json, 跳过 (要重做先删掉)"; exit 0; }
  log "第③步数据"
  python data/convert_sft35_stage3.py --tokenizer "$TOKENIZER" --out "$DATA" "${@:2}" 2>&1 | tee -a "$ROOT/logs/sft35.data3.log"
  ;;
smoke)
  model_of a; need_files "${DATA_FILES[@]}"; need_gpus
  log "第②步冒烟 3 步"
  train_cmd "outputs/${TAG}_smoke" --smoke-steps 3 2>&1 | tee "$ROOT/logs/$TAG.smoke.log" | tail -40
  ;;
train)
  model_of "${2:-}"; need_files "${DATA_FILES[@]}"; need_gpus
  [[ -z $INIT || -f $INIT/adapter_model.safetensors ]] || { log "缺起点补丁 $INIT (先训上一步)"; exit 1; }
  if [[ -f $OUT/adapter_model.safetensors ]]; then log "$TAG 已有 adapter, 跳过训练"; exit 0; fi
  RESUME=(); [[ -f $OUT/checkpoint/state.pt ]] && RESUME=(--resume)
  log "$TAG 训练开始 (起点 ${INIT:-原版}, lr $LR, $EPOCHS 轮) ${RESUME[*]:-}"
  train_cmd "$OUT" ${RESUME[@]+"${RESUME[@]}"} >> "$ROOT/logs/$TAG.train.log" 2>&1
  log "$TAG 训练完成: $(grep -v '"eval"' "$OUT/train_log.jsonl" | tail -1)"
  ;;
merge)
  model_of "${2:-}"
  [[ -f $OUT/adapter_model.safetensors ]] || { log "$TAG 还没有 adapter, 先 train"; exit 1; }
  if [[ -f $MERGED/model.safetensors.index.json ]]; then log "$TAG 已合并, 跳过"; exit 0; fi
  FREE=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
  (( FREE >= 80 )) || { log "磁盘只剩 ${FREE} GB, 合并要约 75 GB, 先清理 (或删掉上一个评测完的合并模型)"; exit 1; }
  need_gpus
  rm -rf "$MERGED"
  log "$TAG 合并到 $MERGED"
  python train/interns2/merge_lora.py --adapter "$OUT" --out "$MERGED" >> "$ROOT/logs/$TAG.merge.log" 2>&1
  log "$TAG 合并完成"
  ;;
eval)
  model_of "${2:-}"
  [[ -f $MERGED/model.safetensors.index.json ]] || { log "$TAG 还没有合并模型, 先 merge"; exit 1; }
  need_gpus
  run_evals "$TAG" "$MERGED"
  ;;
evalzh)
  case "${2:-}" in
    base)  run_tag=interns2_base;    dir=$MODELS/Intern-S2-Preview ;;
    final) run_tag=interns2_mix_300; dir=$MODELS/Intern-S2-Preview-MedLoRA ;;
    six)   run_tag=interns2_6ds;     dir=$ROOT/merged/interns2_6ds ;;
    *) echo "evalzh 只能是 base / final / six"; exit 1 ;;
  esac
  [[ -f $dir/config.json ]] || { log "找不到 $dir (six 要先用 run_6ds.sh merge 合并)"; exit 1; }
  need_gpus
  run_evals "$run_tag" "$dir" zh
  ;;
status)
  for f in "${STAGE2[@]}" "${STAGE3[@]}"; do printf "%-28s %s\n" "$(basename "$f")" "$([[ -f $f ]] && echo 有 || echo 缺)"; done
  for m in a b c; do
    model_of $m
    printf "%-14s adapter:%s 合并:%s 评测:%s\n" "$TAG" "$([[ -f $OUT/adapter_model.safetensors ]] && echo 有 || echo 无)" \
      "$([[ -f $MERGED/model.safetensors.index.json ]] && echo 有 || echo 无)" "$(ls outputs/eval/*_$TAG.json 2>/dev/null | wc -l) 项"
    [[ -f $OUT/train_log.jsonl ]] && echo "    训练: $(grep -v '"eval"' "$OUT/train_log.jsonl" | tail -1)"
  done
  ;;
*)
  echo "用法: $0 data2 | data3 | smoke | train a|b|c | merge a|b|c | eval a|b|c | evalzh base|final|six | status"; exit 1 ;;
esac
