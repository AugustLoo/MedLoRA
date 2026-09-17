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
  llamafactory-cli train configs/bf16/sft_slake_qlora.yaml &&
  bash train/eval_all.sh sft_r16_server outputs/sft_slake_qlora_r16
' > "$LOG" 2>&1
```
Ctrl+B 再按 D 离开会话; 回来 `tmux attach -t ID_train`; `tail -f $LOG` 看进度。
B1 / B2 把 yaml 换成 `cpt_pubmed_qlora` + `sft_after_cpt`、`cpt_iu_qlora` + `sft_after_cpt_iu` 即可 (见 `scripts/run_workstation.sh` 里的顺序)。
`CUDA_VISIBLE_DEVICES` 选哪张卡只对那条命令生效; 代码里一律 cuda:0, 不要写 cuda:1。

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
