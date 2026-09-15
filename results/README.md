# 实验结果汇总

所有数字由 `train/eval_all.sh` 产出, 原始 json 存在本目录, 逐题预测在 `outputs/eval/*_preds.jsonl` (不进 git)。

## 实验 0 · 基座 zero-shot (2026-09-15)

模型 Qwen/Qwen2.5-VL-3B-Instruct, 无 adapter, Kaggle T4, fp16, max_pixels 401408。

| 表 | 数据 | 指标 | 数值 |
|---|---|---|---|
| 医学能力 | SLAKE test en, 1061 题 | closed acc (416 题) | **67.31** |
| | | open EM / recall / F1 (645 题) | 40.62 / 46.73 / 47.53 |
| | 按模态 closed acc | X-Ray / CT / MRI | 80.70 / 64.49 / 56.82 |
| 通用保持 | TextVQA val, 300 题, seed 42 | VQA acc | **83.89** |
| 可靠性 | PubMedQA pqa_labeled, 1000 题 | acc / macro-F1 | **65.80** / 51.03 |
| | 预测分布 yes / no / maybe | 641 / 201 / 158 | 真实 552 / 338 / 110 |

观察:
- SLAKE closed 67% 是起点, 文献里 SFT 后的 3B 级模型通常能到 80%+, 提升空间明确。
- MRI 最弱 (56.8%), X-Ray 最强 (80.7%), 说明基座见过的胸片远多于 MRI, CPT 数据配比要偏向 CT/MRI。
- PubMedQA 明显偏 yes: 该答 no 的 338 题只预测了 201 个 no, 这是「过度自信」的直接证据, 对齐阶段的目标之一。
- TextVQA 83.89 是遗忘对照基线, 之后每个 adapter 都要和这个数比。

SLAKE 全量耗时 2269 s (约 38 min), 三张表合计约 1.5 h。
