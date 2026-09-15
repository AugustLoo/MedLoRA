# Model Card: <adapter 名称>

> 交付物「评估卡 / 模型卡」。每个正式发布的 adapter 填一份, 放在 adapter 目录旁边。

## 基本信息
- 基座模型: Qwen/Qwen2.5-VL-3B-Instruct (commit: ___)
- 训练方式: QLoRA (rank ___, alpha ___, target: all linear, 视觉塔冻结)
- 训练阶段: [ ] 只 SFT  [ ] CPT → SFT  [ ] CPT → SFT → 对齐
- 训练数据: 见 data card ___
- 训练配置: configs/___.yaml (seed ___)
- 硬件与时长: ___ GPU × ___ 小时
- adapter 大小: ___ MB

## 评估 (与基座对比, 同一评估脚本、同一 seed)

| 指标 | 基座 | 本 adapter | Δ |
|---|---|---|---|
| SLAKE closed acc | | | |
| SLAKE open recall | | | |
| SLAKE open EM | | | |
| TextVQA acc (n=300, 通用能力保持) | | | |
| PubMedQA acc / macro-F1 | | | |
| PubMedQA "maybe" 比例 (可靠性) | | | |

按模态拆分 (CT / MRI / X-Ray): ___

## 已知局限
- 只在 SLAKE 英文子集上训练, 不覆盖 ___
- 未做临床验证, 仅供研究, 不得用于实际诊疗
- 遗忘: TextVQA 下降 ___ 个点, 原因分析 ___

## 预期用途与禁止用途
- 用途: 课题 6 实验复现, 供课题 5 / 10 / 11 下游使用
- 禁止: 任何真实临床决策

## 复现
```
llamafactory-cli train configs/___.yaml
bash train/eval_all.sh <tag> outputs/<adapter>
```
