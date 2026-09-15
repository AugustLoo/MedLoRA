# MedLoRA 16 周执行计划

课题: Topic 6 · 医学多模态模型的参数高效增量预训练与指令微调 (任务 1.3)
分工: 本人 Topic 6, 队友 Topic 1 (胸片图文对比学习, 其数据子集与 embedding 接口是本课题的上游)
硬件: 本机 RTX 3050 4 GB 只写代码; 训练与评估在 Kaggle T4×2 (每周 30 h), 冲刺期可租 AutoDL 4090

## 一、总览: 四条线并行

| 线 | 内容 | 依赖 |
|---|---|---|
| 主线 | SLAKE + PubMedQA: 基线 → SFT → CPT+SFT → 消融 | 无, 全程可独立完成 |
| 原型线 | IU X-Ray (Kaggle 公开) 跑通图文 CPT 与报告造题脚本 | 无 |
| 扩展线 | 队友的 CheXpert Plus 子集换入原型脚本, 做图文 CPT 主实验 | 队友第 5 周交子集与切分 |
| 合规线 | PhysioNet (MIMIC-CXR) 与 CheXpert Plus 授权申请 | 第 1 周提交 |

主线独立就能交齐全部四项交付物; 扩展线是加分章节。

## 二、周计划

### 第 1 周: 起点
- [x] 仓库搭建, 推 GitHub (AugustLoo/MedLoRA)
- [x] Kaggle 账号手机验证, API 令牌, notebook 推送
- [ ] 零样本基线三张表 (进行中)
- [ ] 提交 PhysioNet 申请 (CITI 培训 + DUA), 注册 CheXpert Plus
- [ ] 与队友约定: 子集 JSONL 字段、病人级切分共用、交付日期 (第 5 周初子集, 第 6 周初 embedding)
- [ ] 向助教确认: 报告字数/语言/是否答辩
- [ ] 开报告文档, 先写「研究现状」「方法」骨架

### 第 2–4 周: 实验 A (只 SFT)
- SLAKE 训练集转 sharegpt 格式, `configs/sft_slake_qlora.yaml` 跑 QLoRA SFT
- 三张表评估, 与基线对比, 填第一张模型卡
- 逐题错例分析: 基线错在哪, SFT 后修了哪些, 新错了哪些
- 原型线: 挂载 IU X-Ray, 写 `data/convert_iu_xray.py` (图文 CPT 语料 + 报告问答题)
- 里程碑: 第 4 周末有「基线 vs SFT」完整对比表

### 第 5–8 周: 实验 B (CPT → SFT)
- 文本 CPT: PubMedQA 2 万条, `configs/cpt_pubmed_qlora.yaml`
- 图文 CPT: IU X-Ray 原型 → 换入 CheXpert Plus 子集 (5k → 20k)
- 每个 CPT adapter 上接 SFT (`configs/sft_after_cpt.yaml`), 三张表评估
- 用队友的对齐分数做数据过滤, 对比过滤前后 CPT 效果
- 里程碑: 第 8 周末有「A vs B」核心对比表, 这是报告的主结果

### 第 9–12 周: 实验 C/D (遗忘控制与消融)
- C: SFT 时混入通用 VQA 回放数据 (比例 5% / 10%), 看 TextVQA 恢复多少
- D 消融: lora_rank (8/16/32), CPT 数据量 (5k/20k/50k), 学习率, 训练轮数
- 可靠性: PubMedQA maybe 比例分析; 时间允许则做一轮 DPO 对齐
- 里程碑: 第 12 周末所有实验跑完, 数字冻结

### 第 13–16 周: 交付
- 整理 adapter (每个实验一个目录 + 模型卡)
- 数据卡: 每版训练混合一份
- 报告: 研究现状 / 方法 / 实验设置 / 结果与分析 / 局限 / 复现说明
- 仓库整理: README 能让别人一条命令复现任一实验
- 答辩演示 (若有): 一张图输入 → 基线回答 vs 微调回答

## 三、实验矩阵

| 编号 | 配置 | 回答的问题 |
|---|---|---|
| 0 | 基座 zero-shot | 起点 |
| A | SLAKE SFT | 指令微调带来多少提升 |
| B1 | PubMedQA 文本 CPT → SFT | 文本 CPT 有无增益 |
| B2 | 胸片图文 CPT → SFT | 图文 CPT 有无增益 |
| B3 | 对齐分数过滤后的图文 CPT → SFT | 数据质量过滤的价值 (用队友接口) |
| C | B 最优 + 通用回放 | 能否减轻遗忘 |
| D | rank / 数据量 / lr / epoch 消融 | 哪个因素起作用 |
| E (可选) | C + DPO 对齐 | 可靠性能否提升 |

每组都跑 `train/eval_all.sh`, 三个 json 直接填模型卡。

## 四、三张评估表

| 表 | 数据 | 指标 | 对应课题要求 |
|---|---|---|---|
| 医学能力 | SLAKE test (en) | closed acc, open recall/EM/F1, 按模态拆分 | 医学 VQA 提升 |
| 通用保持 | TextVQA val 固定 300 条 | VQA acc | 减缓通用能力退化 |
| 可靠性 | PubMedQA pqa_labeled 1000 条 | acc, macro-F1, maybe 比例 | 幻觉评估 |

## 五、交付物对照 (PDF 原文四项)

| PDF 要求 | 仓库位置 |
|---|---|
| 训练数据生成脚本 | `data/*.py` |
| CPT/SFT 流水线和 adapter | `configs/`, `train/`, `outputs/<exp>/` |
| 评估数据集和模型卡 | `eval/`, `cards/` |
| 实验报告和复现配置 | 报告文档 + `configs/*.yaml` + README |

## 六、与队友的接口

需要他交的:
1. 胸片子集 JSONL: image_path, patient_id, split, findings, impression, labels, view (第 5 周初, 先 5k)
2. 病人级切分文件, 两边共用 (与 1 同时)
3. 图文对齐模型的打分接口: 输入 (图, 文) → 匹配分; 输入图 → 14 标签概率 (第 6 周初)

给他的:
- 微调后的 VLM adapter, 供他做「检索 vs 生成式问答」对比
- 评估指标代码、切分逻辑共用

## 七、红线
- SLAKE test 与 PubMedQA pqa_labeled 只评估, 永不训练
- 训练集与测试集病人无交集, 切分文件固定并入 git
- 受控数据 (MIMIC / CheXpert) 与病人级衍生数据不进 git、不上传公开 Kaggle 数据集
- 训练与评估共用 `medvlm/prompts.py`, 改模板两边同步
- 每个实验一个 yaml, seed 固定, 数字可复现
