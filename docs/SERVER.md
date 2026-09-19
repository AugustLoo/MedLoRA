# 在学院 GPU 服务器上跑 MedLoRA

按老师发的《Member Guide and Usage Rules》整理, 只保留和本项目有关的命令。
下面把个人标识符写成 `ID`, 账号写成 `user0`, 拿到管理员发的账号后替换。

## 0. 先向管理员要
- 账号、密码、个人标识符 (ID)
- 服务器指纹 (首次 ssh 会问, 核对后再接受)
- 显卡型号和显存 (手册没写; 决定用不用 `configs/bf16/`)
- VS Code 连接包 `gpu-ready-user0.zip` (可选, 没有也能用普通 ssh)

## 1. 登录
```bash
ssh -p 20322 user0@221.239.50.147
whoami; cd /workspace; pwd
```

## 2. 首次: 建目录和环境 (只做一次)
```bash
mkdir -p /workspace/ID/MedLoRA /workspace/ID/data /workspace/ID/runs
source /opt/conda/etc/profile.d/conda.sh
conda create --copy --prefix "$HOME/envs/ID_medlora" python=3.11 pip
conda activate "$HOME/envs/ID_medlora"
python -c "import sys; print(sys.executable)"     # 应为 /home/user0/envs/ID_medlora/bin/python

# 服务器指定的 torch 版本组合
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

cd /workspace/ID
git clone https://github.com/AugustLoo/MedLoRA
cd MedLoRA
python -m pip install -r requirements.txt
python -m pip install "llamafactory[torch,metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git"
python -m pip check

# GPU 自检 (先 nvidia-smi 确认逻辑 GPU 0 空闲)
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, 'GB')"
```
数据放 `/workspace/ID/data`: 把 `data/raw` 指过去, 免得占项目目录。
```bash
rm -rf data/raw && ln -s /workspace/ID/data data/raw
python data/download_slake.py && python data/convert_slake_sharegpt.py
python data/download_pubmedqa.py && python data/convert_pubmedqa_cpt.py --max 10000
```
HuggingFace 下载失败时先试 `export HF_ENDPOINT=https://hf-mirror.com`, 仍失败按手册把完整报错发给管理员。

## 3. 每次登录都要做
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/envs/ID_medlora"
cd /workspace/ID/MedLoRA
nvidia-smi                      # 看哪张卡空着, 和同账号的人说一声
```

## 4. 跑实验 (放进 tmux, 断线不停)
```bash
tmux new -s ID_train            # 已存在则 tmux attach -t ID_train
source /opt/conda/etc/profile.d/conda.sh && conda activate "$HOME/envs/ID_medlora" && cd /workspace/ID/MedLoRA
python scripts/make_bf16_configs.py           # 显存 >= 24 GB 用 configs/bf16/, 否则用 configs/ (4bit)

LOG=/workspace/ID/runs/A-$(date +%Y%m%d-%H%M%S).log
CUDA_VISIBLE_DEVICES=0 bash -c '
  CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/bf16/sft_slake_qlora.yaml &&
  bash train/eval_all.sh sft_r16_server outputs/sft_slake_qlora_r16
' > "$LOG" 2>&1
```
Ctrl+B 再按 D 离开会话; 回来 `tmux attach -t ID_train`; `tail -f $LOG` 看进度。
B1 / B2 把 yaml 换成 `cpt_pubmed_qlora` + `sft_after_cpt`、`cpt_iu_qlora` + `sft_after_cpt_iu` 即可 (见 `scripts/run_workstation.sh` 里的顺序)。
`CUDA_VISIBLE_DEVICES` 选哪张卡只对那条命令生效; 代码里一律 cuda:0, 不要写 cuda:1。

## 4b. 看现在有没有在跑

```bash
tmux ls                                              # 自己的会话名以 ID_ 开头; 会话在 != 任务在跑
nvidia-smi                                           # 下半部分 Processes 空的就是没人跑
ps -eo pid,etime,cmd | grep llamafactory | grep -v grep   # 路径带 /workspace/ID/ 的才是自己的, etime 是已跑时长
tmux attach -t ID_train                              # 进去看进度; Ctrl+B 松开再按 D 离开, 任务不停
```

共享账号下别人的进程用户名和自己一样, 只能靠 tmux 会话名和命令行里的 `/workspace/ID/` 路径区分。
训练进度看步数 (`{'loss': ...,  'epoch': ...}` 那几行), 单卡 bf16 的实验 A/C1 总步数应为 1091-1092。

## 5. 结果拿回本机
```bash
scp -P 20322 -r user0@221.239.50.147:/workspace/ID/MedLoRA/outputs/eval ./outputs/eval-server
```
训练中不要覆盖服务器上正在用的代码和数据。checkpoint 跑完及时删 (`outputs/*/checkpoint-*`), 手册要求控制磁盘。

## 6. 规矩 (手册第 7 节)
- 只动自己的目录和环境; 共享账号能看到别人的文件不等于可以改。
- 先 nvidia-smi 再开跑, 优先逻辑 GPU 0; 多卡要事先商量。
- 只停自己的任务, 禁止 `pkill python` / `killall python`。
- 不改密码、共享 `.bashrc`、SSH 设置; 不重启容器, 不装驱动, 不装 CUDA Toolkit。
- 删除没有回收站; 私钥和密码不进聊天和仓库。
- 求助时给: 账号、ID、项目路径、python 路径、命令、完整报错、时间。

## 实验 C1 · SLAKE SFT + PubMedQA 三分类回放

**要回答的问题**：CPT-only 评估显示 yes 偏置是短答式 SFT 引入的（`results/README.md`）。
那么在 SFT 数据里混入三分类样本，maybe 能不能保住，SLAKE 会不会掉。

**数据**：SLAKE train 4,919 条 + PubMedQA 900 条（每类 300，来自 `data/pubmedqa_split.json` 的 train 半边，
maybe 只有 55 条唯一样本，会被重复约 5 次）。test 半边永不进训练，评估只认它。
其余超参与实验 A 完全一致，唯一变量就是这 900 条。

```bash
tmux attach -t chunqian_train     # 没有会话就 tmux new -s chunqian_train
source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian && cd /workspace/chunqian/MedLoRA
python data/convert_pubmedqa_sft.py --per-class 300
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/bf16/sft_mix_pubmedqa.yaml 2>&1 | tail -40
rm -rf outputs/sft_mix_pubmedqa_r16/checkpoint-*
CUDA_VISIBLE_DEVICES=0 bash train/eval_all.sh sft_mix_pubmedqa_r16 outputs/sft_mix_pubmedqa_r16   > /workspace/chunqian/runs/c1.log 2>&1
```

训练约 1 小时，评估约 1 小时（另一账号的 srb 占着算力时会更久）。

**训练必须绑单卡**：`llamafactory-cli` 看到两张卡会自动开分布式，有效 batch 从 16 变 32，
和实验 A 就不止一个变量了（2026-09-19 踩过，步数从 1091 变成 546 是识别信号）。
每条训练命令前面都要有 `CUDA_VISIBLE_DEVICES=0`，这也符合手册「多卡要事先商量」的要求。

**已知风险**：这一轮把带图的 SLAKE 和纯文本的 PubMedQA 混在同一次 SFT 里。
若 LLaMA-Factory 在预处理阶段报缺 `images` 字段，把报错发回来，
退路是分两段训（先 SLAKE 后 PubMedQA），但那样就不是「混合」而是「续训」，结论要另写。

**结果怎么看**：PubMedQA 一律用 test 半边，本机跑
`python scripts/eval_pubmedqa_split.py --half test`，它会把 C1 和之前所有模型放在同一把尺子上。
SLAKE 和 TextVQA 照旧全量，与 `baseline_server` 比。

## 实验 A-server · 把实验 A 在服务器上用 C1 的条件重跑

**为什么要跑**：实验 A 是在 Kaggle 上 4bit QLoRA + fp16 训的，C1 是服务器上不量化 bf16。
两者差了「数据配比 / 量化 / 硬件」三项。校准结论幅度大（macro-F1 +9.4，maybe 5→19）不受影响，
但「SLAKE 无代价」和「TextVQA −2.9」都是一分左右的判断，需要一个单变量对照才站得住。
跑完 A-server 之后，A-server 与 C1 之间唯一的差别就是那 900 条 PubMedQA。

```bash
tmux attach -t chunqian_train     # 没有会话就 tmux new -s chunqian_train
source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian && cd /workspace/chunqian/MedLoRA
nvidia-smi                        # 先确认 GPU 0 空着

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/bf16/sft_slake_qlora.yaml 2>&1 | tail -40
rm -rf outputs/sft_slake_qlora_r16/checkpoint-*
CUDA_VISIBLE_DEVICES=0 bash train/eval_all.sh sft_a_server outputs/sft_slake_qlora_r16   > /workspace/chunqian/runs/a_server.log 2>&1
```

训练约 1 小时（总步数应为 1091，出现 546 就是没绑住单卡，Ctrl+C 重来），评估约 1 小时。

**取回**（本机，注意别让 scp 把文件套进子目录）：
```bash
scp -P 20322 chunqian@221.239.50.147:/workspace/chunqian/MedLoRA/outputs/eval/'*sft_a_server*' outputs/eval/
python scripts/eval_pubmedqa_split.py --half test
```

**看什么**：A-server 与 C1 比 SLAKE 封闭/开放，差值若在一分内则「回放不伤主任务」成立；
A-server 的 TextVQA 若也低于 84.22，说明那 2.9 分是 SFT 本身造成的，不能算在回放头上。

