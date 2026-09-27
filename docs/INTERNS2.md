# Intern-S2-Preview (35B) 上重复回放实验

2026-09-25 起 · 在 8 卡 5090 主机上用管理员账号 `ubuntu` 操作 (`ssh 5090`)

## 分工与规矩

- 模型和推理服务是同学的: `/home/ubuntu/Large-Model-Service-Interns2`。**只读**, 不改、不复制、不移动。
- 我们的东西全部放 `/home/ubuntu/chunqian/`: 代码 `MedLoRA/`、训练环境 `envs/s2train/`、数据 `data/`。
- 显卡: 方案 A。训练时先 `stop_server.sh` 关掉推理服务, 用 0-3 号卡; 6、7 号有别人的 `srb`, 不碰。训练完再 `start_server.sh` 开回来。
- 评估: 基座已经通过接口测完 (results/README.md)。训练好的 adapter 怎么评估, 见文末。

## 一、训练环境 (一次)

同学的推理环境里 torch 2.10+cu128、transformers 5.12、flash-linear-attention 0.5.2 已经在 5090 上验证过。
**复制一份**到我们目录再补装, 不动原环境:

```bash
E=/home/ubuntu/Large-Model-Service-Interns2/envs/inference
ls $E/conda-meta >/dev/null 2>&1 && echo "是 conda 环境, 可以 clone" || echo "不是 conda 环境, 看下面的备用做法"
$E/bin/python -c "import torch, transformers, peft, accelerate; print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)"

mkdir -p /home/ubuntu/chunqian/envs
conda create -y -p /home/ubuntu/chunqian/envs/s2train --clone $E
conda activate /home/ubuntu/chunqian/envs/s2train
pip install -U peft -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
```

备用 (不是 conda 环境时): 新建 `conda create -y -p /home/ubuntu/chunqian/envs/s2train python=3.12`, 再
`pip install -r /home/ubuntu/Large-Model-Service-Interns2/requirements-lock.txt` (torch 要走 cu128 的源)。

## 二、代码和数据

代码: 本机 Git Bash 打包上传, 服务器上解压 (**必须带 `--keep-directory-symlink`**, 见 SERVER.md):

```bash
# 本机
cd "/c/Users/Y4/other projects/Topic 6/MedLoRA" && scp 5090-update.tar.gz 5090:/home/ubuntu/chunqian/
# 服务器
mkdir -p /home/ubuntu/chunqian/MedLoRA && tar -xzf /home/ubuntu/chunqian/5090-update.tar.gz -C /home/ubuntu/chunqian/MedLoRA --keep-directory-symlink
```

数据: 训练文件 (`slake_train.json`、`pubmedqa_sft_train_300.json`) 和 SLAKE 图片在 user0 容器里,
主机上看不到, 要拷到 `/home/ubuntu/chunqian/data/`。训练文件里的图片路径是容器路径, 训练时用 `--path-map` 换成主机路径, 不改文件本身。

## 三、冒烟测试 (先跑这个)

```bash
bash /home/ubuntu/Large-Model-Service-Interns2/scripts/stop_server.sh     # 方案 A: 先关服务
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv     # 0-3 号应为空

conda activate /home/ubuntu/chunqian/envs/s2train && cd /home/ubuntu/chunqian/MedLoRA
CUDA_VISIBLE_DEVICES=0,1,2,3 python train/interns2/train_lora.py \
  --data /home/ubuntu/chunqian/data/processed/slake_train.json /home/ubuntu/chunqian/data/processed/pubmedqa_sft_train_300.json \
  --path-map /workspace/chunqian/MedLoRA/data/raw/SLAKE=/home/ubuntu/chunqian/data/SLAKE \
  --path-map /workspace/chunqian/data/SLAKE=/home/ubuntu/chunqian/data/SLAKE \
  --output outputs/interns2_smoke --smoke-steps 3 2>&1 | tee outputs/interns2_smoke.log
```

要看四样: LoRA 挂了多少层、可训练参数占比; 两条样本的提示词里有没有图片标记、是不是关思考的格式; 计损失的是不是只有答案
(比如 `'No<|im_end|>'`); 每步秒数和各卡显存峰值。

## 四、正式训练

两轮, 和 3B 对应:

| 轮 | 数据 | 对应 3B |
|---|---|---|
| `interns2_mix_0` | 只有 `slake_train.json` | 回放 0 (A-server) |
| `interns2_mix_300` | `slake_train.json` + `pubmedqa_sft_train_300.json` | 回放 300 |

命令同冒烟测试, 去掉 `--smoke-steps 3`, 改 `--output`, 在 tmux 里跑。断了加 `--resume` 从最近的检查点 (每 100 步一个) 接着跑。
进度看 `outputs/<名字>/train_log.jsonl` 最后一行, 里面有预计剩余分钟数。

## 五、评估训练好的模型

LoRA 挂在线性注意力的投影层上, LMDeploy 能否直接加载这种 adapter 未验证。两条路, 冒烟和训练跑通后再定:

1. 把 adapter 合并进权重, 存一份完整模型到我们目录 (约 69 GB), 用同学的 `start_server.sh` 换模型路径起服务, 走接口评估。
2. 用 transformers 本地加载「基座 + adapter」在 0-3 号卡上直接生成, 评估脚本走本地模式 (慢, 但不依赖 LMDeploy)。

## 记录 (2026-09-25)

- 训练环境: 克隆同学的 inference 环境后, peft 升到 0.21, **transformers 必须降到 5.2.0** (模型自带代码按 5.2 写,
  5.12 里 `create_causal_mask` 不收 `cache_position`); 不指定 `attn_implementation` (时间序列子模块不支持 sdpa)。
- 冒烟: LoRA 挂 250 个线性层, 可训练 19,169,280 / 35,270,223,168 = 0.054%; 标签只覆盖答案 (`Head<|im_end|>`、`maybe<|im_end|>`);
  batch 2 时显存峰值 16.9 / 18.1 / 18.1 / 20.4 GiB, 首步 83 秒 (编译), 之后约 11 秒一步。
- 正式训练用 batch 4 × 累积 4 (与 3B 相同):

| 轮 | 步数 | 用时 | 秒/步 | 末段 loss | 超长跳过 |
|---|---|---|---|---|---|
| interns2_mix_300 | 978 | 1 h 58 min | 5.75 | 0.053 | 0 |
| interns2_mix_0 | 921 | 1 h 18 min | 4.97 | 0.029 | 0 |

- 评估走「合并 → 起服务 → 接口」: `train/interns2/merge_lora.py` 把 adapter 并进权重存到 `/home/ubuntu/chunqian/merged/<名字>`,
  `train/interns2/serve.sh start <目录>` 用同学环境里的 LMDeploy 在 23334 端口起服务 (参数与他的 start_server.sh 相同),
  容器里 `MEDVLM_API_BASE=http://172.17.0.1:23334/v1` 跑四张表, 与基座评估路径完全一致。

## 结果 (2026-09-26)

三个模型都已评估完, 数字与结论在 `results/README.md` 最后两节。一句话: 3B 的四条结论在 35B 上全部复现 ——
SFT 压掉 maybe (幅度小得多)、回放 300 修复 (且不过冲)、主任务零代价、代价只在短答格式。
合并后的两份完整模型在 `/home/ubuntu/chunqian/merged/` (共约 146 GB), 需要时可删, adapter 在 `outputs/` 里, 随时能重新合并。

## D 组消融 (35B, 2026-09-26 起)

在回放 300 的设置上每次只改一样, 对照组是已跑过的 `interns2_mix_300`:

| 名字 | 改动 | 问题 |
|---|---|---|
| ep1 | 3 轮 → 1 轮 | 是否根本不用练 3 轮 |
| attn | LoRA 只挂注意力, 不挂共享专家 | 混合专家模型特有: 共享专家的 LoRA 有没有用 |
| r8 / r32 | rank 16 → 8 / 32 (alpha 同比) | 容量 |
| lr5e-5 / lr2e-4 | 学习率减半 / 加倍 | 步长 |

`train/interns2/ablation.sh` 全自动跑完一轮的全部步骤 (训练 → 合并 → 服务 → 评估 → 停服务 → 删合并模型), 评估也在主机上做,
不用在两个窗口之间切换。可重复运行, 做完的自动跳过。每轮约 3 小时 (ep1 约 1.5 小时), 6 轮约 15-17 小时, 期间同学的服务停着。

**一次性准备 (ubuntu 主机):**

```bash
conda activate /home/ubuntu/chunqian/envs/s2train
pip install datasets scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import transformers, datasets, sklearn; print(transformers.__version__, datasets.__version__)"   # transformers 必须仍是 5.2.0

# 评估数据: 三个 HF 数据集的缓存从 user0 容器拷到主机 (放我们自己的 HF_HOME, 不碰 ~/.cache)
mkdir -p /home/ubuntu/chunqian/hf
ssh -p 20322 user0@127.0.0.1 "du -sh /home/user0/.cache/huggingface/hub/datasets--*"
ssh -p 20322 user0@127.0.0.1 "tar -C /home/user0/.cache/huggingface -cf - hub/datasets--qiaojin--PubMedQA hub/datasets--lmms-lab--textvqa hub/datasets--lmms-lab--MMBench" | tar -C /home/ubuntu/chunqian/hf -xf -

# SLAKE 测试集: 让主机上的 data/raw/SLAKE 指到已经拷过来的图片
cd /home/ubuntu/chunqian/MedLoRA && ln -sfn /home/ubuntu/chunqian/data/SLAKE data/raw/SLAKE && ls data/raw/SLAKE/test.json
```

**开跑:** 停同学的服务, 在 tmux 里 `bash train/interns2/ablation.sh`。进度 `tail -3 /home/ubuntu/chunqian/logs/ablation.log`,
每轮的训练 / 合并 / 服务 / 评估日志在同目录 `interns2_abl_<名字>.*.log`。全部跑完后 `start_server.sh` 把同学的服务开回来。

## 消融结果 (2026-09-27)

6 轮全部跑完 (ep1 训练 41 分钟; 其余五轮每轮连合并和评估约 2 小时 20 分), 结果与结论在 `results/README.md` 最后一节和报告 5.7 节。
坑: 第一次在主机上评估时只拷了 HF 下载缓存, `datasets` 离线读不到 TextVQA; 还要拷 `~/.cache/huggingface/datasets/` 下的处理后缓存
(`lmms-lab___textvqa`、`qiaojin___pub_med_qa`、`lmms-lab___mm_bench`)。换机器评估前先用三行 `load_dataset` 测一遍再开跑。
`ablation.log` 里的 PubMedQA F1 是全部 1000 题口径 (含训练半边), 比较时用 `scripts/eval_pubmedqa_split.py` 的考卷半边数字。

## 补种子与开放式探针 (2026-09-27)

回放 0、回放 300、只挂注意力、只练 1 轮各两个种子; 13 个 35B 模型都跑了开放式探针 (`eval/eval_openended.py`)。
35B 种子间 macro-F1 差可达 3.7 (3B 约 0.7), 单次运行不可靠。结论见 `results/README.md` 最后一节与报告 5.7 节。
开放式探针的「异常说成正常」在 44 张图上太不稳 (同一条件两个种子 13 与 3), 不作为指标使用。

