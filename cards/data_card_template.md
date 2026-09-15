# Data Card: <数据集 / 训练混合名称>

> 交付物「数据卡」。每一版训练数据配比填一份。

## 来源
| 子集 | 来源 | 许可 | 用途 (CPT / SFT / 评估) | 条数 |
|---|---|---|---|---|
| SLAKE en train | github.com/SuperJohnZhang/Slake | CC BY 4.0 | SFT | |
| SLAKE en validation | 同上 | | 训练中 eval | |
| SLAKE en test | 同上 | | **只评估, 不训练** | |
| PubMedQA pqa_artificial | github.com/pubmedqa/pubmedqa | MIT | CPT | |
| PubMedQA pqa_labeled | 同上 | | **只评估, 不训练** | |
| MIMIC-CXR (可选) | physionet.org | PhysioNet 受控 | CPT | |

## 处理步骤
1. 语言过滤: 只保留 q_lang == en
2. 提示词模板: medvlm/prompts.py (训练与评估共用)
3. 图像: 原图, image_max_pixels = 262144
4. 去重 / 泄漏检查: 训练集与测试集 qid 交集 = ___ (必须为 0)

## 配比
| 实验 | SLAKE | PubMedQA CPT | 通用回放数据 | 备注 |
|---|---|---|---|---|
| A 只 SFT | 100% | 0 | 0 | |
| B CPT→SFT | | 20k | 0 | |
| C CPT→SFT+回放 | | | ___ | 缓解遗忘 |

## 合规
- MIMIC 系列: 已完成 PhysioNet CITI 培训 (日期 ___), DUA 签署 (日期 ___)
- 原始受控数据与病人级衍生数据不进 git, 不上传公开仓库
- 本仓库 .gitignore 已排除 data/raw, data/processed, outputs

## 已知偏差
- SLAKE 模态分布不均 (CT/MRI/X-Ray 各 ___)
- closed 题 yes/no 比例 ___, 可能导致模型偏向某一答案
