# S2-6datasets 设计（2026-10-08）

老师要求在 Intern-S2-Preview（35B）上新训两个模型，保留原有的 S2-2datasets。本文件只覆盖 **S2-6datasets**；
**S2-CheXpertPlus 暂缓**，要等「描述漏报」改用 CheXbert 判定（下周第一项）之后再做，否则量不出它有没有用。

## 1. 三个模型

| 名字 | 内部标签 | 训练数据 | 状态 |
|---|---|---|---|
| S2-2datasets | `interns2_mix_300` | SLAKE 4,919 + PubMedQA 300 | 已有，交给世龙部署的那个，不动 |
| **S2-6datasets** | `interns2_6ds` | 下表六个数据集的训练部分，约 17,000 条 | 本设计 |
| S2-CheXpertPlus | `interns2_chexpert` | CheXpert Plus 胸片 → 报告 | 暂缓 |

训练方法与 S2-2datasets 完全相同（`train/interns2/train_lora.py` 默认值）：LoRA r16 / alpha 32 / dropout 0.05、
注意力 + 共享专家、lr 1e-4、3 轮、有效 batch 16、截断 1024、图片 ≤ 262,144 像素、种子 42、关闭思考模式。
**只有数据不同**，这样三个模型之间的比较才是单变量。先训一个种子，结果值得写进报告时再补种子 43。

## 2. 数据（只用训练部分，测试部分永远只考试）

| 数据集 | 来源 | 训练部分 | 用量 | 提示词（与评测相同） | 答案 |
|---|---|---|---|---|---|
| SLAKE | 现有 `slake_train.json` | 4,919 | 全部 | `slake_prompt` | 原答案 |
| PubMedQA | 现有 `pubmedqa_sft_train_300.json` | 300 | 全部 | `PUBMEDQA` | yes / no / maybe |
| VQA-RAD | `flaviagiammarino/vqa-rad` train | 1,793（待核） | 全部 | 金标 yes/no → `SLAKE_CLOSED`，其余 → `SLAKE_OPEN` | 原答案 |
| PathVQA | `flaviagiammarino/path-vqa` train | 19,654（待核） | 抽 5,000 | 同 VQA-RAD | 原答案 |
| MedQA | `GBaker/MedQA-USMLE-4-options` train | 10,178（待核） | 抽 3,000 | `MEDQA`（选项 `A. …` 每行一个） | 字母 |
| PneumoniaMNIST 224 | `danjacobellis/pneumoniamnist_224` train | 4,708（待核） | 抽 2,000，肺炎 / 正常各 1,000 | `SLAKE_CLOSED`「Does this chest X-ray show pneumonia?」 | yes / no |

- 抽样数量都是命令行参数（`--pathvqa 5000` 等），以后要加大，改数字重新生成、重新训练即可。抽样用固定种子 42，结果可复现。
- 抽样理由：全量约 41,000 条，PathVQA 与 MedQA 会占七成，淹没主任务 SLAKE；训练时间也会超过两天。
- 「待核」的条数在服务器上下载后打印核对，与公开说明不符就停下来报告。

### 防泄漏检查（转换脚本自动做，结果写进数据卡）

1. **题目重叠**：训练部分与测试部分按「图片哈希 + 问题文本」比对，重叠的训练样本剔除并计数。
2. **图片重叠**：VQA-RAD 已知同一张图在训练和测试部分都出现（问题不同）。统计重叠图片数，
   在报告里注明：S2-6datasets 在 VQA-RAD 上的分数是「领域内、且见过部分测试图片」。
3. PneumoniaMNIST 训练 / 测试是不同图片，同样按像素哈希核对一遍；MedQA 按题干文本核对。

### 存放

- 训练文件：`data/processed/{vqarad,pathvqa,medqa,pneumonia}_sft_train.json`（不进 git）。
- 图片：`data/processed/images/{vqarad,pathvqa,pneumonia}/`（不进 git），训练文件里写主机上的绝对路径。
- 全部在 5090 主机上下载和生成，走 hf-mirror，不经过本机网络。

## 3. 训练与合并

- 新脚本 `train/interns2/run_6ds.sh`：检查六个数据文件都在 → 3 步冒烟 → 正式训练 → 合并。
- 输出 adapter `outputs/interns2_6ds/`；合并后的完整模型放 `/home/ubuntu/chunqian/merged/interns2_6ds/`（约 73 GB，
  开工前先查磁盘空间）。之后可加进对比接口（`compare_serve.sh switch six`）。
- 预计 3,190 步左右，按 S2-2datasets 每步约 7 秒，**6–8 小时**；要占 0-3 号卡，和世龙约一个晚上。

## 4. 评测

与 S2-2datasets 完全相同的整套，结果三列并排（原版 / S2-2datasets / S2-6datasets）：

- SLAKE 测试、PubMedQA 考卷半边、TextVQA 300、MMBench 500；
- VQA-RAD / PathVQA / MedQA / PneumoniaMNIST 测试部分 —— **对 S2-6datasets 标注为「领域内」**，不再算外部泛化；
- CheXpert Plus 3,189 张漏报测试（容器里跑，`serve_for_container.sh` 加 `six`）。

## 5. 文档

数据卡加新的训练来源与许可（VQA-RAD CC0 1.0、PathVQA MIT、MedQA CC BY 4.0、MedMNIST CC BY 4.0）和防泄漏统计；
模型卡加 `interns2_6ds`；报告新增一节；结果快照 `results/interns2_6ds_<日期>.json`。

## 6. 不做的事

- 不动 S2-2datasets 和世龙目录里的任何模型。
- 不碰 CheXpert Plus（S2-CheXpertPlus 暂缓）。
- 不加新的训练技巧（不改学习率、不加权重采样），保持单变量。
