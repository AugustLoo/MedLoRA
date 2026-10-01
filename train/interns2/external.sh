#!/usr/bin/env bash
# 外部测试集 (2026-09-30 老师同意): VQA-RAD / PathVQA / MedQA, 加上「漏报异常」探针 PneumoniaMNIST (624 张),
# 只测不训, 在 35B 的三个模型上各跑一遍。PneumoniaMNIST 测试集要先从本机上传到 data/raw/medmnist/ (见 docs/INTERNS2.md)。
#   base   : 基座               /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
#   final  : 最终版 (回放 300) /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview-MedLoRA (已合并好, 只读使用)
#   r0     : 只做微调 (回放 0)  outputs/interns2_mix_0 临时合并到 $ROOT/merged/, 评完就删
#   r0_s43 / s43 : 回放 0 与回放 300 的第二个种子 (outputs/interns2_abl_r0_s43 / interns2_abl_s43), 同样临时合并, 用来复核新发现
# 标签与之前的评估一致 (interns2_base / interns2_mix_300 / interns2_mix_0), 结果可以和 SLAKE / PubMedQA 直接放一起看。
#
# 用法 (ubuntu 主机):
#   bash train/interns2/external.sh prefetch        # 第一步: 走镜像下载三个测试集 (只下测试分片, 约 170 MB), 不占卡, 不用停服务
#   bash train/interns2/external.sh base final r0   # 第二步: 在 tmux 里, 先停同学的服务, 依次评测; 做完的自动跳过
# 进度: tail -5 /home/ubuntu/chunqian/logs/external.log
# 全部跑完后, 同学的服务要由他 (或经他同意) 重新启动。
set -Eeuo pipefail

ROOT=/home/ubuntu/chunqian
REPO=$ROOT/MedLoRA
MODELS=/home/ubuntu/Large-Model-Service-Interns2/models
LOG=$ROOT/logs/external.log
mkdir -p "$ROOT/logs" "$ROOT/merged"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/s2train"
cd "$REPO"
export HF_HOME=$ROOT/hf OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [[ "${1:-}" == prefetch ]]; then
  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
  python - <<'PY'
from datasets import load_dataset
for repo, files in [("flaviagiammarino/vqa-rad", "data/test-*.parquet"),
                    ("flaviagiammarino/path-vqa", "data/test-*.parquet"),
                    ("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_test.jsonl")]:
    ds = load_dataset(repo, data_files={"test": files}, split="test", verification_mode="no_checks")
    print(repo, len(ds), "题 OK")
PY
  exit 0
fi

# 评测阶段也保持联网走镜像: 用 data_files 只下测试分片时, 离线模式下 datasets 算出的缓存配置编号与下载时不同,
# 会报 Couldn't find cache (datasets 的已知问题, 2026-10-01 踩到)。文件都已缓存, 联网只发元数据请求, 不会重新下载。
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
cleanup() { bash "$REPO/train/interns2/serve.sh" stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

[[ $# -gt 0 ]] || { echo "用法: $0 prefetch | base final r0 r0_s43 s43"; exit 1; }
[[ -f data/raw/medmnist/pneumoniamnist_224_test.npz ]] || { echo "缺 data/raw/medmnist/pneumoniamnist_224_test.npz, 先从本机上传"; exit 1; }
for NAME in "$@"; do
  TMP_MERGED=""
  case $NAME in
    base)  TAG=interns2_base;    DIR=$MODELS/Intern-S2-Preview ;;
    final) TAG=interns2_mix_300; DIR=$MODELS/Intern-S2-Preview-MedLoRA ;;
    r0)    TAG=interns2_mix_0;   DIR=$ROOT/merged/interns2_mix_0; TMP_MERGED=$DIR ;;
    r0_s43) TAG=interns2_abl_r0_s43; DIR=$ROOT/merged/$TAG; TMP_MERGED=$DIR ;;
    s43)   TAG=interns2_abl_s43;  DIR=$ROOT/merged/$TAG; TMP_MERGED=$DIR ;;
    *) echo "未知模型 $NAME (可选 base / final / r0 / r0_s43 / s43)"; exit 1 ;;
  esac
  if [[ -f outputs/eval/vqarad_$TAG.json && -f outputs/eval/pathvqa_$TAG.json && -f outputs/eval/medqa_$TAG.json && -f outputs/eval/pneumonia_$TAG.json ]]; then
    log "$NAME ($TAG): 四个测试都已有结果, 跳过"; continue
  fi
  log "==== $NAME ($TAG) ===="
  if [[ -n $TMP_MERGED && ! -f $TMP_MERGED/model.safetensors.index.json ]]; then
    rm -rf "$TMP_MERGED"
    log "合并 outputs/$TAG"
    python train/interns2/merge_lora.py --adapter outputs/$TAG --out "$TMP_MERGED" >> $ROOT/logs/$TAG.merge.log 2>&1
  fi
  [[ -f $DIR/config.json ]] || { log "找不到模型目录 $DIR"; exit 1; }
  log "起服务 $DIR"
  bash train/interns2/serve.sh start "$DIR" >> $ROOT/logs/$TAG.serve.log 2>&1
  for DS in vqa-rad path-vqa; do
    OUTN=${DS//-/}
    [[ -f outputs/eval/${OUTN}_$TAG.json ]] && { log "$DS 已有, 跳过"; continue; }
    log "$DS 开始"
    python eval/eval_medvqa.py --dataset $DS --model "$DIR" --tag $TAG >> $ROOT/logs/$TAG.external.log 2>&1
    log "$DS 完成: $(python -c "import json;m=json.load(open('outputs/eval/${OUTN}_$TAG.json'));x=m['metrics'];y=m['closed_yes_no'];print('封闭',x['closed_acc'],'开放EM',x['open_em'],'| 预测yes%',y['pred_yes_pct'],'真实yes%',y['gold_yes_pct'],'no→yes',y['no_to_yes'],'yes→no',y['yes_to_no'])")"
  done
  if [[ ! -f outputs/eval/medqa_$TAG.json ]]; then
    log "MedQA 开始"
    python eval/eval_medqa.py --model "$DIR" --tag $TAG >> $ROOT/logs/$TAG.external.log 2>&1
    log "MedQA 完成: $(python -c "import json;m=json.load(open('outputs/eval/medqa_$TAG.json'));print('准确率',m['medqa_acc'],'未解析',m['unparsed'])")"
  fi
  if [[ ! -f outputs/eval/pneumonia_$TAG.json ]]; then
    log "PneumoniaMNIST 开始 (是非题 + 自由描述)"
    python eval/eval_pneumonia.py --model "$DIR" --tag $TAG --freetext >> $ROOT/logs/$TAG.external.log 2>&1
    log "PneumoniaMNIST 完成: $(python -c "import json;m=json.load(open('outputs/eval/pneumonia_$TAG.json'));c,f=m['closed'],m['freetext'];print('灵敏度',c['sensitivity'],'特异度',c['specificity'],'漏报',c['missed_pneumonia'],'| 描述里肺炎片没报肺部异常',f['pneu_no_lung_abnormality'],'/',m['n_pneumonia'],'| 是非题答yes但描述没报',f['closed_yes_but_desc_no_lung_pct'],'%')")"
  fi
  bash train/interns2/serve.sh stop >> $ROOT/logs/$TAG.serve.log 2>&1
  if [[ -n $TMP_MERGED ]]; then rm -rf "$TMP_MERGED"; log "已删除临时合并模型"; fi
done
log "全部完成: $*。记得请同学重新启动他的服务。"
