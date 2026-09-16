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

## 实验 A · SLAKE QLoRA SFT (2026-09-15)

配置 `configs/sft_slake_qlora.yaml`: Qwen2.5-VL-3B, 4bit, rank 16, lr 1e-4, 3 epoch, 有效 batch 16, Kaggle T4 单卡。
训练 4919 条 (SLAKE en train), 4 h 28 min, 0.92 样本/秒; 训练中验证 4 次各 7 min。adapter 120 MB。
train loss 0.736 → 0.082; eval loss 0.225 → 0.163 → 0.149 → 0.145 (仍在下降, 3 epoch 未过拟合); 无 NaN。

| 指标 | 基座 | 实验 A | Δ |
|---|---|---|---|
| SLAKE closed acc | 67.31 | **89.18** | +21.87 |
| SLAKE open EM / recall / F1 | 40.62 / 46.73 / 47.53 | **75.35 / 82.17 / 81.56** | +34.7 / +35.4 / +34.0 |
| SLAKE closed by modality X-Ray / CT / MRI | 80.70 / 64.49 / 56.82 | 91.23 / 86.45 / 93.18 | +10.5 / +22.0 / +36.4 |
| TextVQA acc (遗忘对照) | 83.89 | **83.89** | 0.00 |
| PubMedQA acc / macro-F1 | 65.80 / 51.03 | 70.60 / 52.38 | +4.8 / +1.4 |
| PubMedQA 预测 yes / no / maybe | 641 / 201 / 158 | 685 / 256 / 59 | 真实 552 / 338 / 110 |

按题型 (准确率, open 题为 EM):

| 题型 | n | 基座 | A | Δ |
|---|---|---|---|---|
| Position open | 163 | 24.5 | 57.7 | +33.1 |
| Organ open | 99 | 23.2 | 84.8 | +61.6 |
| KG open | 109 | 22.0 | 72.5 | +50.5 |
| Abnormality open | 41 | 12.2 | 41.5 | +29.3 |
| Abnormality closed | 109 | 68.8 | 82.6 | +13.8 |
| Organ closed | 154 | 75.3 | 90.9 | +15.6 |
| Modality / Plane / Color / Size closed | 91 | 26.9–78.6 | 100.0 | — |

逐题: 修正 345 题, 新错 30 题。

观察:
- SFT 收益主要来自「词表对齐」: Organ/KG/Color/Plane 这些基座已经看懂但用词不对的题型涨幅最大 (+50 到 +60)。
- 真正的医学识别题 Abnormality open 只到 41.5, Position open 只到 57.7, 这两类是 CPT 和数据配比要攻的地方; 剩余错例多为左右/上下定位错误和病灶类别混淆。
- MRI 从最弱 (56.8) 变最强 (93.2): SLAKE 训练集里 MRI 题多, 说明基座缺的是领域数据而非能力。
- TextVQA 分数完全不变 (300 题中 81 题输出仅大小写不同, 归一化后相同): LoRA + 冻结视觉塔在这个对照上没有可测的遗忘。下一步要换一个更敏感的通用基准 (如 MMBench 子集或开放式描述题) 再确认。
- PubMedQA 的 yes 偏置没有改善, maybe 从 158 降到 59 (真实 110): SFT 让模型更「敢答」, 可靠性反而略降, 这是对齐阶段的靶子。
- 训练时间瓶颈: 0.92 样本/秒。下一轮把训练中验证改为每 400 步且只抽 200 题, 可省约 25 min。

## 实验 B1 · PubMedQA 文本 CPT → SLAKE SFT (2026-09-16)

配置 `configs/cpt_pubmed_qlora.yaml` (1 万条 pqa_artificial, 1 epoch, lr 5e-5, packing 1024) → `configs/sft_after_cpt.yaml` (从 CPT adapter 继续, 与实验 A 同参数)。
CPT 2 h 39 min (0.42 样本/秒, loss 1.96 → 1.75); SFT 4 h 07 min (loss 1.38 → 0.079, eval 0.160 → 0.142); 评估 1.5 h; 合计 8.4 h。无 NaN。

| 指标 | 基座 | A (只 SFT) | B1 (CPT→SFT) | B1−A |
|---|---|---|---|---|
| SLAKE closed acc | 67.31 | 89.18 | 88.70 | −0.48 |
| SLAKE open EM / recall / F1 | 40.62 / 46.73 / 47.53 | 75.35 / 82.17 / 81.56 | 75.81 / 82.72 / 82.07 | +0.5 / +0.6 / +0.5 |
| SLAKE closed X-Ray / CT / MRI | 80.70 / 64.49 / 56.82 | 91.23 / 86.45 / 93.18 | 91.23 / 85.51 / 93.18 | 0 / −0.9 / 0 |
| TextVQA acc | 83.89 | 83.89 | 83.78 | −0.11 |
| PubMedQA acc / macro-F1 | 65.80 / 51.03 | 70.60 / 52.38 | **72.80 / 54.19** | +2.2 / +1.8 |
| PubMedQA 预测 yes / no / maybe | 641 / 201 / 158 | 685 / 256 / 59 | 670 / 286 / 44 | 真实 552 / 338 / 110 |

按题型 B1−A: Position open +2.5, Quantity +3.8, KG open −2.8, Abnormality open −2.4, 其余 ±1 以内。逐题 A→B1: 修正 15 题, 新错 14 题。

结论:
- **纯文本 CPT 对图像 VQA 没有可测增益**: SLAKE 全部指标与 A 的差异在 ±1 内, 逐题互换 15/14, 属随机波动。原因是 CPT 语料 (文献摘要) 与 SLAKE (影像定位/病灶识别) 的知识不重叠, 且 CPT 只动 LLM 部分, 视觉塔冻结。
- **CPT 的收益出现在它自己的领域**: PubMedQA +2.2 (文本推理), 说明 CPT 本身是有效的, 只是没有跨模态迁移。
- **yes 偏置仍在**: no→yes 误判 92 例, maybe→yes 77 例; maybe 只答了 44 个。文本 CPT 没有改善可靠性。
- 由此实验 B2 (图文 CPT, IU X-Ray / CheXpert Plus 的图片+报告) 是必要的, 否则 CPT 阶段对本课题的主指标无贡献。报告中 B1 作为「文本 CPT 不足以提升医学 VQA」的对照证据保留。
- 时间: CPT 的 packing 序列在 T4 上只有 0.42 样本/秒, 1 万条已经 2.6 h; 图文 CPT 数据量要控制在 5k 以内或等学院服务器。
