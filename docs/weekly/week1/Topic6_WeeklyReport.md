# Topic 6 · Week 1 进展（补记）

> 第 1 周没有正式汇报，这页是按仓库记录补的，方便和后面的周报对齐。

**做了什么**
- 确定课题：任务 1.3，2B–7B 开源 VLM 的 CPT → LoRA/QLoRA SFT → 对齐；与队友分工（队友 Topic 1，胸片图文对比学习）。
- 选基座 Qwen2.5-VL-3B-Instruct，训练框架 LLaMA-Factory；本机 4 GB 显存只做代码，训练与评估上 Kaggle T4。
- 搭好仓库 AugustLoo/MedLoRA：数据脚本、三张评估表（SLAKE / TextVQA-300 / PubMedQA）、统一提示词、一个实验一个 yaml。
- 跑通零样本基线：SLAKE closed 67.3 / open recall 46.7，TextVQA 83.9，PubMedQA 65.8（yes 偏置明显）。

**发现**
- 基座 X 光最好、MRI 最差，说明缺的是领域数据。
- PubMedQA 该答 no 的 338 题只答了 201 个 no，是「过度自信」的直接证据。

**下一周**
- 实验 A（只 SFT）与实验 B（CPT → SFT）。
