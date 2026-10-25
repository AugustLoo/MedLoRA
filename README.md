# MedLoRA · 医学多模态模型的参数高效增量预训练与指令微调

Parameter-efficient continued pre-training and instruction tuning of medical vision-language models.

本科毕业设计 (四川大学), 作者 LOO CHUN QIAN, 导师徐蕾。
研究问题: 在看图 + 文字的大模型上, 只训练很少一部分参数 (LoRA), 「知识预训练 → 问答微调 → 推理训练」的每一步各带来多少提升; 微调后模型描述医学图像时可不可靠。

> 代码助手 (Codex 等) 请先读 [AGENTS.md](AGENTS.md): 服务器规矩、数据红线和一步步的操作命令都在那里。

## 两个阶段

| 阶段 | 模型 | 硬件 | 状态 |
|---|---|---|---|
| 一 (2026-09) | Qwen2.5-VL-3B-Instruct, QLoRA, LLaMA-Factory | Kaggle T4 / 单张 24-32 GB 卡 | 已结束 (`configs/`、`notebooks/`、`train/run_*.sh`) |
| 二 (2026-09 起) | Intern-S2-Preview 35B 混合专家, LoRA (r 16, 只训 0.054% 参数), 自写训练循环 | 4 × RTX 5090 (学院服务器) | 主线 (`train/interns2/`) |

## 主要结论 (详见 [docs/report.md](docs/report.md))

- **分数来自问答微调**: 35B 在 SLAKE 是非题上 84.4 → 94.2, 开放题也大幅提升; 去掉图像后是非题掉到约 50% (瞎猜水平), 说明模型真在看图。
- **知识预训练没有可测提升**: 3B 和 35B 都是这样。35B 用 1,940 万 token 的教材、指南、百科做预训练, 配对 bootstrap 区间下除 PathVQA (−0.8) 外没有一项变化; 换种子 43 重训, 结论不变。
- **推理训练 (13,000 道带解析的考题) 也没有可测提升**, 只有儿童肺炎题 +1.3。
- **少量回放数据能修正「不敢答 maybe」的偏差**: 混入 300 条三分类 PubMedQA, 35B 的 macro-F1 61.4 → 66.3, 主任务不受影响。
- **可靠性问题**: 在 3,189 张成人胸片上, 微调后的模型是非题更愿意说「有异常」, 但自由描述里更常把异常片说成正常, 两种问法互相矛盾。这个漏报率换随机种子就能差约 4 点, 单次训练的结论不可靠。

所有数字的出处: [results/README.md](results/README.md) (实验日志) 和 `results/*.json` (快照)。

## 目录

```
train/interns2/   35B 训练、合并、起服务、评测 (每个脚本开头的注释就是用法)
scripts/server/   服务器运维: 借卡时间窗 (gpu_window.sh)、容器评测、语料下载
eval/             各测试的评测脚本 (经 OpenAI 兼容接口调用模型)
data/             数据下载与转换; 原始数据和处理结果不进 git
medvlm/           共用代码: 提示词、指标、接口调用
scripts/          汇总、配对区间、图、报告 PDF
results/          结果快照 + 实验日志
docs/             英文报告、模型卡 / 数据卡、35B 方案与操作记录、接口说明
tests/            pytest (数据转换和训练脚本的纯逻辑)
configs/ notebooks/   阶段一 (3B) 的配置和 Kaggle notebook
```

## 评测

| 方面 | 测试 | 脚本 |
|---|---|---|
| 医学看图问答 | SLAKE (英 / 中)、VQA-RAD、PathVQA | `eval/eval_slake.py`、`eval/eval_medvqa.py` |
| 医学考题 | MedQA、CMExam、PubMedQA (固定考卷半边) | `eval/eval_medqa.py`、`eval/eval_cmexam.py`、`eval/eval_pubmedqa.py` |
| 通用能力保持 | TextVQA 300 题、MMBench 500 题 | `eval/eval_general.py`、`eval/eval_mmbench.py` |
| 可靠性 | 去掉图像对照、儿童肺炎描述漏报、成人胸片漏报 (CheXpert Plus, 关键词 + CheXbert) | `eval/eval_slake.py --no-image`、`eval/eval_pneumonia.py`、`eval/eval_chexpert.py`、`eval/chexbert_judge.py` |

模型差异用配对 bootstrap 95% 区间判断 (`scripts/bootstrap_ci.py`), 区间不跨 0 才算有差别。

## 复现

- 35B 主线: [AGENTS.md](AGENTS.md) 第 4 节 (服务器上一步步的命令) 和 [docs/report.md](docs/report.md) 第 8 节。
- 三步训练的方案和数据来源: [docs/PLAN_35B_CPT.md](docs/PLAN_35B_CPT.md)。
- 阶段一 (3B) 在单卡上复现:

```bash
pip install -r requirements.txt
pip install "llamafactory[torch,metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git"
python data/download_slake.py && python data/convert_slake_sharegpt.py
bash train/eval_all.sh baseline
llamafactory-cli train configs/sft_slake_qlora.yaml && bash train/eval_all.sh sft_r16 outputs/sft_slake_qlora_r16
```

## 数据与合规

- 公开数据 (SLAKE、PubMedQA、VQA-RAD、PathVQA、MedQA、CMExam、MedMCQA、PneumoniaMNIST、HuatuoGPT / DISC-Med 对话、OpenMedText、临床指南、MMedC) 按各自许可使用, 见 [docs/cards/DATA_CARD.md](docs/cards/DATA_CARD.md)。
- **CheXpert Plus 是受控数据** (签署 DUA): 图片、报告和逐题记录只留在服务器, 仓库里只有汇总比例。
- 训练好的 LoRA 补丁由作者存档, 不公开发布。模型卡: [docs/cards/MODEL_CARD.md](docs/cards/MODEL_CARD.md)。
