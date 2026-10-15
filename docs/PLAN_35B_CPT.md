# 35B 增量预训练 + 指令微调方案（8 个数据集）

- 负责人：Chunqian（Topic 6 / Task 1.3）
- 日期：2026-10-15
- 状态：**方案，尚未开始**（等老师确认、等 0–3 号卡空出来）
- 相关：现有三个 35B 模型的训练细节见 `docs/TRAINING_35B.md`

---

## 一、代码在哪里

- GitHub：<https://github.com/AugustLoo/MedLoRA>（看不到的话告诉我，我加你为协作者）
- 服务器（5090 主机，`ubuntu` 账号）：`/home/ubuntu/chunqian/MedLoRA`

| 用途 | 文件 |
|---|---|
| 35B LoRA 训练（自写，transformers + peft） | `train/interns2/train_lora.py` |
| 补丁合并成完整模型 | `train/interns2/merge_lora.py` |
| 一条龙脚本（数据 → 冒烟 → 训练 → 合并 → 评测） | `train/interns2/run_6ds.sh`（S2-6datasets）、`train/interns2/run_cxp.sh`（S2-CheXpertPlus） |
| 四模型对比服务（端口 23334） | `train/interns2/compare_serve.sh` |
| 六个数据集的微调数据 | `data/convert_sft6.py`（+ `convert_slake_sharegpt.py`、`convert_pubmedqa_sft.py`） |
| CheXpert Plus 胸片 → 报告数据 | `data/convert_chexpert_s2.py`（35B）、`data/convert_chexpert_b3.py`（3B 的 B3） |
| PubMed 摘要预训练语料 | `data/convert_pubmedqa_cpt.py` |
| IU X-Ray 预训练语料 | `data/convert_iu_xray.py` |
| 评测 | `eval/eval_slake.py`、`eval_pubmedqa.py`、`eval_general.py`（TextVQA）、`eval_mmbench.py`、`eval_medvqa.py`（VQA-RAD / PathVQA）、`eval_medqa.py`、`eval_pneumonia.py`、`eval_chexpert.py`、`chexbert_judge.py` |
| 3B 预训练的配置（参照） | `configs/bf16/cpt_pubmed_qlora.yaml`、`cpt_iu_qlora.yaml`、`cpt_b3_top.yaml`、`cpt_b3_rand.yaml`、`sft_after_cpt*.yaml` |
| 已训好的补丁（服务器） | `outputs/interns2_6ds/`、`outputs/interns2_cxp/`、`outputs/interns2_mix_300/` |

---

## 二、LoRA 设置（和现有 35B 模型完全相同）

| 项目 | 设置 |
|---|---|
| 基座 | Intern-S2-Preview（35B 混合专家，256 个路由专家、每个词激活 8 个，约 3B 激活），思考模式关闭 |
| 方法 | LoRA：rank 16，alpha 32（缩放 2），dropout 0.05 |
| 挂在哪里 | 语言模型的注意力层（全注意力 q/k/v/o、线性注意力 in_proj_qkv / in_proj_z / out_proj）+ 共享专家（gate / up / down），共 250 个线性层 |
| 可训练参数 | 19,169,280，占 **0.054%** |
| 冻结 | 256 个路由专家（打包成 3D 参数，挂不了 LoRA）、路由器、视觉编码器、时间序列模块、词表 |
| 代码位置 | `train_lora.py` 里的 `LORA_TARGETS["attn_shared"]` |

---

## 三、预训练的逻辑

**预训练（CPT）**：先让模型读大量医学材料，熟悉这个领域的语言和知识，不教它答题。
**指令微调（SFT）**：再给它「问题 → 标准答案」，教它按要求回答。

两个阶段**用同一个 LoRA 补丁接着训**，全程只训练 0.054% 的参数（这就是「参数高效」），也不用多合并一个 73 GB 的模型：

```
原版 35B
   │  阶段 1：增量预训练（2 个数据集，学习率 5e-5，1 轮）
   ▼
预训练补丁 ──────────────▶ 「只做预训练」模型（顺带评测）
   │  阶段 2：指令微调（6 个数据集，学习率 1e-4，3 轮，从预训练补丁接着训）
   ▼
预训练 + 微调补丁 ──合并──▶ 完整模型 ──▶ 评测，和 S2-6datasets（只微调）对比
```

两种语料的训练方式和 3B 时一致：

- **纯文字**（PubMed 摘要）：整段文字都算损失（3B 的 B1，LLaMA-Factory `stage: pt`）。
- **图文**（CheXpert Plus 胸片 + 报告）：沿用 LLaVA-Med 第一阶段的做法，给图和一句固定的「写报告」提示，只对报告文字算损失（3B 的 B2 / B3）。

---

## 四、8 个数据集

| 阶段 | 数据集 | 数量 | 类型 |
|---|---|---|---|
| 预训练 | PubMed 摘要（PubMedQA `pqa_artificial`，不含考试用的 `pqa_labeled`） | 10,000 | 纯文字 |
| 预训练 | CheXpert Plus 胸片 → 报告（与 S2-CheXpertPlus 同一批 1 万个病人，与 3,189 张测试片无重叠；受控数据，只在服务器上用） | 10,000 | 图文 |
| 微调 | SLAKE | 4,919 | 看图问答 |
| 微调 | PubMedQA（回放） | 300 | 是 / 否 / 不确定 |
| 微调 | VQA-RAD（已去掉问测试图的 1,059 题） | 734 | 看图问答 |
| 微调 | PathVQA | 5,000 | 病理图问答 |
| 微调 | MedQA | 3,000 | 医师考试四选一 |
| 微调 | PneumoniaMNIST | 2,000 | 儿童胸片判断肺炎 |

微调部分与 S2-6datasets **完全相同**（15,953 条），这样唯一的变量就是「有没有先预训练」。

**待确认**：预训练的两个数据集。备选是 IU X-Ray（3,483 对胸片报告，3B 的 B2 用过）。CheXpert Plus 也可以先用 Rafi（项目 1）的对齐分数筛选，再经 Redoan（项目 2）的统一格式打包后使用。

---

## 五、要比较的模型

| 模型 | 预训练 | 微调 | 状态 |
|---|---|---|---|
| 原版 | 无 | 无 | 已有 |
| S2-6datasets | 无 | 6 个数据集 | 已有（只做微调的对照组） |
| **只做预训练** | 2 个数据集 | 无 | 新（阶段 1 的产物，不用另外训练） |
| **预训练 + 微调** | 2 个数据集 | 6 个数据集 | 新，主实验 |

回答的问题：在 35B 上，先做增量预训练，再做同样的指令微调，会不会比直接微调更好？3B 上做过同样的比较（B1 / B2 / B3：都没有提升），两个规模可以互相印证。

---

## 六、训练设置

| | 阶段 1：预训练 | 阶段 2：微调 |
|---|---|---|
| 学习率 | 5e-5（比微调低，减轻遗忘；与 3B 相同） | 1e-4 |
| 轮数 | 1 | 3 |
| 有效批大小 | 16 | 16 |
| 长度上限 | 1,024 token | 1,024 token |
| 图片 | ≤ 262,144 像素（512×512） | 同左 |
| 学习率曲线 | 余弦，3% 预热 | 余弦，5% 预热 |
| 随机种子 | 42 | 42 |
| 起点 | 原版 | 阶段 1 的补丁 |

---

## 七、要改的代码

1. `train_lora.py` 加 `--init-adapter`：阶段 2 从阶段 1 的补丁开始训（现在的 `--resume` 只能续同一次训练）。
2. `train_lora.py` 加纯文字预训练模式：整段文字都算损失（现在只算答案部分）。
3. 新数据脚本：PubMed 摘要 1 万条 + CheXpert Plus 1 万对，转成训练格式（复用 `convert_pubmedqa_cpt.py`、`convert_chexpert_s2.py` 和 B3 的提示词）。
4. 新的一条龙脚本 `run_cpt.sh`（照 `run_6ds.sh` 写）：数据 → 冒烟 → 阶段 1 → 阶段 2 → 合并 → 评测。

---

## 八、评测

- **预训练 + 微调**：和 S2-6datasets 完全相同的一整套：SLAKE、PubMedQA、TextVQA、MMBench、VQA-RAD、PathVQA、MedQA、PneumoniaMNIST、CheXpert Plus 3,189 张成人胸片（漏报、误报、自相矛盾），再用 CheXbert 复核。
- **只做预训练**：没学过按要求答题，短答题分数会很低，这是正常的。重点看胸片描述（CheXpert Plus）和 SLAKE。

---

## 九、时间和资源

| 步骤 | 估计 |
|---|---|
| 阶段 1 预训练（约 2 万条 × 1 轮 ≈ 1,250 步） | 约 2–3 小时 |
| 阶段 2 微调（2,991 步，与 S2-6datasets 相同） | 约 5 小时 40 分 |
| 合并（每个模型） | 3–5 分钟 |
| 评测（每个模型） | 约半天 |
| CheXbert 复核（容器里只能用 CPU） | 每个模型约 11 小时 |

- 显卡：主机 0–3 号卡，训练时要先和世龙协调，暂停他的服务。
- 硬盘：一次只放得下一个合并好的新模型（约 73 GB），两个新模型要轮流合并、评测、删除；补丁（约 77 MB）永久保留。

---

## 十、预期和风险

- 结果可能和 3B 一样，看不出提升。两个规模都没提升，本身就是有说服力的结论，可以写成「系统验证了增量预训练在这种条件下的作用」。
- 每个模型只训练一次。如果结论要写进论文，关键的「预训练 + 微调」模型最好再用种子 43 训一次。
