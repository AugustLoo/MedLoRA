# 35B 三步训练方案：知识预训练 → 问答微调 → 推理训练

- 负责人：Chunqian（Topic 6 / Task 1.3）
- 日期：2026-10-15 起草；2026-10-16 改为三步走（苏学长建议），第一步的语料已定
- 状态：**方案，尚未开始**（等老师确认、等 0–3 号卡空出来）
- 相关：现有三个 35B 模型的训练细节见 `docs/TRAINING_35B.md`

---

## 一、代码在哪里

- GitHub：<https://github.com/AugustLoo/MedLoRA>
- 服务器（5090 主机，`ubuntu` 账号）：`/home/ubuntu/chunqian/MedLoRA`

| 用途 | 文件 |
|---|---|
| 35B LoRA 训练（自写，transformers + peft） | `train/interns2/train_lora.py` |
| 补丁合并成完整模型 | `train/interns2/merge_lora.py` |
| 一条龙脚本（数据 → 冒烟 → 训练 → 合并 → 评测） | `train/interns2/run_6ds.sh`（S2-6datasets）、`train/interns2/run_cxp.sh`（S2-CheXpertPlus） |
| 四模型对比服务（端口 23334） | `train/interns2/compare_serve.sh` |
| 六个数据集的微调数据 | `data/convert_sft6.py`（+ `convert_slake_sharegpt.py`、`convert_pubmedqa_sft.py`） |
| 3B 时的预训练语料（参照） | `data/convert_pubmedqa_cpt.py`、`data/convert_iu_xray.py`、`data/convert_chexpert_b3.py` |
| 评测 | `eval/eval_slake.py`、`eval_pubmedqa.py`、`eval_general.py`（TextVQA）、`eval_mmbench.py`、`eval_medvqa.py`（VQA-RAD / PathVQA）、`eval_medqa.py`、`eval_pneumonia.py`、`eval_chexpert.py`、`chexbert_judge.py` |
| 3B 预训练的配置（参照） | `configs/bf16/cpt_pubmed_qlora.yaml`、`cpt_iu_qlora.yaml`、`cpt_b3_top.yaml`、`cpt_b3_rand.yaml`、`sft_after_cpt*.yaml` |
| 已训好的补丁（服务器） | `outputs/interns2_6ds/`、`outputs/interns2_cxp/`、`outputs/interns2_mix_300/` |

---

## 二、LoRA 设置（和现有 35B 模型完全相同）

| 项目 | 设置 |
|---|---|
| 基座 | Intern-S2-Preview（35B 混合专家，256 个路由专家、每个词激活 8 个，约 3B 激活） |
| 方法 | LoRA：rank 16，alpha 32（缩放 2），dropout 0.05 |
| 挂在哪里 | 语言模型的注意力层（全注意力 q/k/v/o、线性注意力 in_proj_qkv / in_proj_z / out_proj）+ 共享专家（gate / up / down），共 250 个线性层 |
| 可训练参数 | 19,169,280，占 **0.054%** |
| 冻结 | 256 个路由专家（打包成 3D 参数）、路由器、视觉编码器、时间序列模块、词表 |
| 代码位置 | `train_lora.py` 里的 `LORA_TARGETS["attn_shared"]` |

三步都在**同一个 LoRA 补丁上接着训**，全程只训练 0.054% 的参数（题目里的「参数高效」），不用多合并模型。

---

## 三、三步的逻辑

```
原版 35B
   │  ① 知识预训练：医学教材 + 临床指南（纯文字，整段算损失）
   ▼
知识补丁 ───────────▶ 「只做预训练」模型（顺带评测）
   │  ② 问答微调：医学问答（中文）+ 六个多模态看图问答数据集（只算答案的损失）
   ▼
问答补丁 ───────────▶ 模型 B
   │  ③ 推理训练：带解释的医学题（学会一步步推理再作答）
   ▼
推理补丁 ───────────▶ 模型 C
```

| 步骤 | 目的 | 属于论文题目的哪部分 |
|---|---|---|
| ① 知识预训练 | 先让模型有医学基本认知 | 增量预训练（Incremental Pretraining） |
| ② 问答微调 | 学会按要求回答医学问题，包括看图答题 | 指令微调（Instruction Fine-Tuning） |
| ③ 推理训练 | 学会解释、一步步推理 | 指令微调（带解释的指令数据） |

为什么预训练只用纯文字：3B 上两次图文预训练（B2 IU X-Ray、B3 CheXpert Plus）都没帮上主任务，B2 只学会了报告写法；纯文字预训练（B1 PubMed）让纯文字医学题提高了 2.2 分。看图能力靠第②步的多模态数据练。

---

## 四、第①步：知识预训练语料（已定，2026-10-16 在 Hugging Face 上核实）

| 语料 | 内容 | 语言 | 全部规模 | 许可 | 本方案用多少 |
|---|---|---|---|---|---|
| **OpenMedText**（`ywchoi/OpenMedText`） | 29 本开源医学教材（解剖、生理、药理、细胞生物学、流行病学等）+ 12.1 万篇 MDPI 期刊文章 | 英文 | 教材全文 33.5 MB | 期刊文章 CC BY 4.0；教材按本分 CC BY / BY-SA / BY-NC / BY-NC-SA | **29 本教材全部**（约 800 万 token） |
| **MMedC**（`Henrychur/MMedC`） | 多语种医学语料：教材、过滤后的网页、医学网站、小数据集 | 8 种语言 | 255 亿 token；中文 47.4 亿 token（其中**教材 12.1 亿**），中文压缩包 4.6 GB | CC BY-NC-SA 4.0 | **中文教材部分抽样**（约 800 万 token） |
| **Clinical Guidelines**（`epfl-llm/guidelines`） | 3.7 万篇临床诊疗指南（CCO、CDC、CMA、ICRC、NICE、PubMed、SPOR、WHO、WikiDoc），Meditron 训练语料的一部分 | 英文 | 878 MB | Common Crawl 使用条款 | **抽样**（约 400 万 token） |

- 合计约 2,000 万 token，切成 1,024 token 一段，约 2 万段，1 轮。
- MMedC 的**英文教材没有包含**（官方说因为版权去掉了），所以英文教材用 OpenMedText，中文教材用 MMedC，正好互补。
- 三个语料都有非商用条款，本项目只做学术研究，论文里注明来源和许可证；处理后的数据不对外再分发。
- 下载：OpenMedText 很小，本机即可；MMedC 中文包 4.6 GB、指南 878 MB，尽量在服务器上直接下载，只解压教材部分（服务器硬盘约 88% 满）。
- 引用：MMedC — Qiu et al., *Towards Building Multilingual Language Model for Medicine*, arXiv 2402.13963；Guidelines — Chen et al., *MediTron-70B*, arXiv 2311.16079。
- 可选：混入 10–20% 通用文字一起训，减少遗忘（文献里最常用的做法）。

---

## 五、第②步：问答微调数据

| 数据 | 数量 | 类型 |
|---|---|---|
| **Huatuo26M-Lite**（`FreedomIntelligence/Huatuo26M-Lite`，Apache-2.0，共 17.8 万条，每条有问题、答案、科室、相关疾病、质量分） | 抽 10,000（优先 5 分、各科室均匀） | 中文医学问答 |
| SLAKE | 4,919 | 看图问答 |
| PubMedQA（回放） | 300 | 是 / 否 / 不确定 |
| VQA-RAD（已去掉问测试图的 1,059 题） | 734 | 看图问答 |
| PathVQA | 5,000 | 病理图问答 |
| MedQA | 3,000 | 医师考试四选一 |
| PneumoniaMNIST | 2,000 | 儿童胸片判断肺炎 |

- 多模态的六个数据集**必须保留**：模型是多模态的，主要考试也是看图问答，第①③步基本是纯文字。
- 华佗数据说明：Huatuo-26M 的 2,600 万条里 2,534 万条是问诊记录，只公开了答案网址、没有文字；能直接用的约 134 万条，所以用质量最高的精选版 Lite。引用：Li et al., *Huatuo-26M*, arXiv 2305.01526。

---

## 六、第③步：推理训练数据（候选，待定）

| 数据 | 内容 | 许可 |
|---|---|---|
| `FreedomIntelligence/medical-o1-reasoning-SFT` | 训练 HuatuoGPT-o1 用的医学推理链数据，中英文 | Apache-2.0 |
| `openlifescienceai/medmcqa` | 印度医学考试选择题，每题带解释 | Apache-2.0 |

**待定：训练时开不开思考模式。** 现在训练和评测都关闭了思考模式，推理数据是长篇推理过程。建议两种都先小规模试，评测也两种都测。推理链很长，长度上限要从 1,024 提到 2,048–4,096，显存和时间都会增加。

参考：第 3 周原版开思考，21% 的题在 4,096 token 内想不完；文献里也有胸片报告的研究发现显式思考没有帮助。推理训练的效果预计主要体现在纯文字题（MedQA、PubMedQA）上。

---

## 七、要比较的模型

| 模型 | ① 知识预训练 | ② 问答微调 | ③ 推理训练 | 回答的问题 |
|---|---|---|---|---|
| 原版 | — | — | — | 基线 |
| S2-6datasets（已有） | — | 只有六个多模态数据集 | — | 参考：没有华佗问答时 |
| **A** | — | ✅ | — | 新对照组（第②步数据和 S2-6datasets 不同，要重训） |
| **B** | ✅ | ✅ | — | 知识预训练有没有用（B 对比 A） |
| **C** | ✅ | ✅ | ✅ | 推理训练有没有用（C 对比 B） |
| 只做预训练 | ✅ | — | — | 顺带评测（第①步的产物） |

3B 上做过类似的预训练对比（B1 / B2 / B3：主任务都没有提升），两个规模可以互相印证。

---

## 八、训练设置

| | ① 知识预训练 | ② 问答微调 | ③ 推理训练 |
|---|---|---|---|
| 学习率 | 先短跑扫描 5e-5 / 1e-4 / 2e-4 再定 | 1e-4 | 待定（建议从 5e-5 起） |
| 轮数 | 1 | 3 | 1–2 |
| 有效批大小 | 16 | 16 | 16 |
| 长度上限 | 1,024 | 1,024 | 2,048–4,096 |
| 损失 | 整段文字 | 只算答案 | 只算推理 + 答案 |
| 图片 | 无 | ≤ 262,144 像素（512×512） | 无 |
| 起点 | 原版 | ① 的补丁（A 从原版） | ② 的补丁 |
| 随机种子 | 42 | 42 | 42 |

---

## 九、要改的代码

1. `train_lora.py` 加 `--init-adapter`：从上一步的补丁开始训（现在的 `--resume` 只能续同一次训练）。
2. `train_lora.py` 加纯文字预训练模式：整段文字都算损失，长文切成 1,024 token 一段。
3. 新数据脚本：
   - 第①步：OpenMedText 教材 + MMedC 中文教材抽样 + 临床指南抽样，清洗后切段，固定种子；
   - 第②步：Huatuo26M-Lite 抽 1 万条，转成「问题 → 答案」，和六个数据集合并；
   - 第③步：推理数据转成训练格式（按是否开思考，分两种模板）。
4. 新的一条龙脚本 `run_3stage.sh`（照 `run_6ds.sh` 写）：数据 → 冒烟 → ① → ② → ③ → 合并 → 评测。
5. `eval/eval_slake.py` 加 `--lang zh`：现在只测英文题，中文题的提示词和「是 / 否」判分要另外处理。

---

## 十、评测

- **A / B / C**：和 S2-6datasets 完全相同的一整套：SLAKE、PubMedQA、TextVQA、MMBench、VQA-RAD、PathVQA、MedQA、PneumoniaMNIST、CheXpert Plus 3,189 张成人胸片（漏报、误报、自相矛盾），再用 CheXbert 复核。
- **新加**：
  - SLAKE 中文测试题 1,033 道，看中文能力；
  - 去掉图像只给文字再测一次，证明模型真的在看图；
  - 关键指标给出置信区间。
- **只做预训练**：短答题分数会很低，这是正常的，重点看纯文字医学题。

---

## 十一、时间和资源（估计）

| 步骤 | 估计 |
|---|---|
| ① 知识预训练（约 2 万段 × 1 轮 ≈ 1,250 步） | 约 2–3 小时，学习率扫描另加几次短跑 |
| ② 问答微调（约 2.6 万条 × 3 轮 ≈ 4,900 步） | 约 9 小时（A、B 各一次） |
| ③ 推理训练（长度长，每步更慢） | 约 3–5 小时 |
| 合并（每个模型） | 3–5 分钟 |
| 评测（每个模型） | 约半天；A、B、C 共三个 |
| CheXbert 复核（容器里只能用 CPU） | 每个模型约 11 小时 |

- 显卡总共约 25–30 小时，用主机 0–3 号卡，训练前要和世龙协调暂停他的服务。
- 硬盘：一次只放得下一个合并好的模型（约 73 GB），要轮流合并、评测、删除；补丁（约 77 MB）永久保留。

---

## 十二、预期和风险

- 知识预训练可能和 3B 一样看不出提升。两个规模都没提升，本身就是有说服力的结论。
- 推理训练的提升预计主要在纯文字题；看图题不一定变好。
- 每个模型只训练一次。关键模型如果要写进论文，最好再用种子 43 训一次。
- 语料的非商用许可：仅限学术研究，论文中注明。
