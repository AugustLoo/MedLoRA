# MedLoRA · 医学多模态模型的参数高效增量预训练与指令微调

课题任务 1.3: 对一个 2B–7B 的开源视觉语言模型做 CPT → LoRA/QLoRA SFT → 对齐,
在 SLAKE 医学 VQA 上提升, 同时控制通用能力遗忘, 产出 adapter + 数据卡 + 模型卡。

## 硬件现实
本机 RTX 3050 只有 4 GB 显存, 只用来写代码和跑 `scripts/smoke_local.ps1` 冒烟测试。
所有正式训练和评估在云端跑: 实验室 GPU > Kaggle (免费 T4×2) > AutoDL 4090。
`notebooks/kaggle_baseline.ipynb` 可以直接上传到 Kaggle 跑基线。

## 目录
```
medvlm/      共用代码: 模型加载 (model.py), SLAKE 读取 (slake.py), 提示词 (prompts.py), 指标 (metrics.py)
data/        下载与转换脚本; raw/ 和 processed/ 不进 git
configs/     LLaMA-Factory yaml, 一个实验一个文件
train/       run_sft.sh / run_cpt.sh / eval_all.sh
eval/        三张评估表: eval_slake (医学) / eval_general (遗忘) / eval_pubmedqa (可靠性)
notebooks/   Kaggle 基线 notebook
cards/       数据卡、模型卡模板
outputs/     adapter、日志、评估结果; 不进 git
```

## 快速开始 (云端)
```bash
pip install -r requirements.txt
pip install "llamafactory[torch,metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git"

python data/download_slake.py          # 检查输出里有 train.json / imgs
python data/download_pubmedqa.py

bash train/eval_all.sh baseline        # 路线第 1 步: zero-shot 基线, 三张表
bash train/run_sft.sh                  # 实验 A: 只 SFT
bash train/eval_all.sh sft_r16 outputs/sft_slake_qlora_r16
bash train/run_cpt.sh                  # 实验 B: CPT → SFT
bash train/eval_all.sh cpt_sft_r16 outputs/sft_after_cpt_r16
```

## 实验矩阵 (对应课题路线 26–30)
| 编号 | 配置 | 回答的问题 |
|---|---|---|
| 0 | 基座 zero-shot | 起点在哪 |
| A | SLAKE SFT | 指令微调带来多少提升 |
| B | PubMedQA CPT → SLAKE SFT | 小规模 CPT 有没有增益 |
| C | B + 通用数据回放 | 能否减轻遗忘 |
| D | 数据配比 / rank 消融 | 哪个因素起作用 |

每个实验都跑 `train/eval_all.sh`, 三个 json 直接填进 `cards/model_card_template.md` 的表。

## 三张评估表
| 表 | 脚本 | 数据 | 指标 | 对应课题要求 |
|---|---|---|---|---|
| 医学能力 | eval/eval_slake.py | SLAKE test (en) | closed acc, open recall/EM/F1, 按模态拆分 | 医学 VQA 提升 |
| 通用保持 | eval/eval_general.py | TextVQA val 固定 300 条 | VQA acc | 减缓通用能力退化 |
| 可靠性 | eval/eval_pubmedqa.py | PubMedQA pqa_labeled 1000 条 | acc, macro-F1, maybe 比例 | 幻觉评估 |

## 数据合规
- SLAKE (CC BY 4.0) 与 PubMedQA (MIT) 可直接用。
- MIMIC-CXR 需要 PhysioNet 授权 (CITI 培训 + DUA + 审核, 1–3 周), 拿到后作为 CPT 扩展语料。
- 受控数据和病人级衍生数据永远不进 git, `.gitignore` 已经排除。

## 注意
- `medvlm/slake.py` 兼容官方 json+imgs 布局和 HF datasets 两种格式。第一次下载后看一眼打印出的文件列表, 确认是哪一种。
- T4 不支持 bf16, 配置里用 fp16。换到 4090/A100 时把 `fp16: true` 改成 `bf16: true`, 并可去掉 `quantization_bit`。
- 训练和评估用同一套提示词 (medvlm/prompts.py), 改模板必须两边一起改。
