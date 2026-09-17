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

## 实验 B2 · IU X-Ray 图文 CPT → SLAKE SFT (2026-09-16)

配置 `configs/cpt_iu_qlora.yaml` (3483 张正位胸片 → Findings+Impression 报告, caption 式对齐, 1 epoch, lr 5e-5) → `configs/sft_after_cpt_iu.yaml` (与 A 同参数)。
CPT 1 h 19 min (0.73 样本/秒, loss 2.17 → 1.06); SFT 4 h 38 min (loss 0.73 → 0.083, eval 0.168 → 0.151); 评估 1.5 h; 合计 7.7 h。无 NaN。

| 指标 | 基座 | A | B1 文本 CPT | B2 图文 CPT | B2−A |
|---|---|---|---|---|---|
| SLAKE closed acc | 67.31 | 89.18 | 88.70 | 88.70 | −0.48 |
| SLAKE open EM / recall / F1 | 40.62 / 46.73 / 47.53 | 75.35 / 82.17 / 81.56 | 75.81 / 82.72 / 82.07 | 74.57 / 81.61 / 80.88 | −0.8 / −0.6 / −0.7 |
| SLAKE closed X-Ray / CT / MRI | 80.70 / 64.49 / 56.82 | 91.23 / 86.45 / 93.18 | 91.23 / 85.51 / 93.18 | **92.11** / 85.05 / 93.18 | +0.9 / −1.4 / 0 |
| TextVQA acc | 83.89 | 83.89 | 83.78 | 84.00 | +0.11 |
| PubMedQA acc / macro-F1 | 65.80 / 51.03 | 70.60 / 52.38 | 72.80 / 54.19 | 70.80 / 52.96 | +0.2 / +0.6 |
| PubMedQA 预测 yes / no / maybe | 641 / 201 / 158 | 685 / 256 / 59 | 670 / 286 / 44 | 672 / 263 / 65 | 真实 552 / 338 / 110 |

按题型 B2−A: Abnormality open **−9.8** (41.5 → 31.7, n=41, 即少对 4 题), Modality closed −3.0, KG closed −2.6, KG open −1.8; Abnormality closed +1.8, Quantity +1.9, Size open +2.6; 其余 ±1。逐题 A→B2: 修正 11 题, 新错 18 题。X 光片子集: open 76.5 → 74.5, closed 91.2 → 92.1。

CPT 阶段冒烟 (给 3 张验证片写报告): 模型完全学会了放射报告的句式和结构 ("The heart is normal in size. The lungs are clear. No pneumothorax or pleural effusion. Impression: No acute cardiopulmonary abnormality."), 但 3 张里 2 张有异常 (双下肺斑片影 / 右侧胸腔积液) 都被写成正常。

结论:
- **图文 CPT 同样没有给 SLAKE 带来增益**, 整体略降 (open −0.6, 逐题 11 对 18)。X 光片封闭题小涨 0.9 (114 题涨 1 题), 在噪声内。
- **CPT 学到的是语言, 不是视觉**: 报告语体学得很像, 但对异常视而不见, 输出向「正常」模板坍缩。IU X-Ray 训练集里 36% 是正常报告, 且异常报告的句子也大多以否定句 ("no effusion") 开头, caption 式 loss 让模型学会的是高频模板。视觉塔冻结, 视觉特征没有被改动, 所以 CPT 不可能提升「看片」。
- 这也解释了 Abnormality open 的下降: 一个偏向「正常」的先验在识别病灶的题上是负作用。
- 与 B1 合起来的结论: **在 3B 模型、冻结视觉塔、几千条数据的设定下, CPT 阶段对医学 VQA 主指标没有贡献**; 文本 CPT 只在文本任务 (PubMedQA) 上有效。报告主结果是 A, B1/B2 作为「CPT 何时无效」的对照证据。
- 后续如果还要做 CPT, 必须改两处之一: 解冻视觉塔 (或给 ViT 也加 LoRA), 或换成异常均衡的图文对 (CheXpert Plus 按标签抽样)。否则把精力转向 SFT 数据配比 (C)、遗忘基准与可靠性对齐。

## 补充 · 只做 CPT、不做 SFT 的 adapter 与基座对比 (2026-09-17, Kaggle 只评估 2.9 h)

| 指标 | 基座 | B1-CPT only | B2-CPT only | A (只 SFT) |
|---|---|---|---|---|
| SLAKE closed acc | 67.31 | 69.95 | 70.43 | 89.18 |
| SLAKE open EM / recall / F1 | 40.62 / 46.73 / 47.53 | 34.11 / 44.87 / 43.35 | 36.43 / 44.65 / 45.49 | 75.35 / 82.17 / 81.56 |
| SLAKE closed X-Ray / CT / MRI | 80.70 / 64.49 / 56.82 | 76.32 / 66.82 / 69.32 | 72.81 / 69.63 / 69.32 | 91.23 / 86.45 / 93.18 |
| TextVQA acc | 83.89 | 81.67 | 83.56 | 83.89 |
| PubMedQA acc / macro-F1 | 65.80 / 51.03 | 66.70 / 53.25 | 68.40 / 53.20 | 70.60 / 52.38 |
| PubMedQA 预测 yes / no / maybe | 641 / 201 / 158 | 578 / 251 / 171 | **598 / 288 / 114** | 685 / 256 / 59 |

(真实分布 552 / 338 / 110)

观察:
- CPT 本身不是没有变化: 封闭题 +2.6 / +3.1, PubMedQA +0.9 / +2.6, 但幅度小, 而且被随后的 SFT 完全盖掉 (SFT 后四个模型都收敛到 89 / 82)。
- 开放题 EM 下降 (40.6 → 34.1 / 36.4) 而 recall 基本不变 (46.7 → 44.9 / 44.7): 是答题格式跑偏 (答成句子), 不是知识变少。B2-CPT 学了写报告, 格式影响最明显。
- **最重要的发现在可靠性**: B2-CPT only 的 PubMedQA 预测分布 598 / 288 / 114 是五个模型里最接近真实分布 552 / 338 / 110 的; B1-CPT only 也把 no 从 201 提到 251。**yes 偏置和 maybe 消失是 SLAKE SFT 引入的**, 不是 CPT 引入的, 更不是基座固有的。对齐阶段的靶子因此更清楚: 问题出在短答式 SFT 数据, 先改 SFT 的数据配比 (混入 PubMedQA 三分类样本), 再考虑 DPO。
- B1-CPT only 的 TextVQA 掉 2.2 (83.9 → 81.7), 是四组里唯一可测的通用退化, SFT 之后又回到 83.8: 文本 CPT 对通用能力有轻微漂移, 探针不是完全测不出。
- 按模态: CPT 后 MRI 封闭题从 56.8 涨到 69.3 (两种 CPT 一样), X-Ray 反而降 (80.7 → 76.3 / 72.8), 需要逐题看是格式还是内容。
