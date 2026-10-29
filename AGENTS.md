# AGENTS.md · 给代码助手的工作说明

这份文件写给在本仓库里干活的代码助手 (Codex 等)。人看的介绍在 [README.md](README.md)。
**动手之前先读完第 1 节「红线」。** 本机若有 `AGENTS.local.md` (不进 git, 写的是个人偏好、联系人和近期事务), 也先读它。
**每做完一件事都要更新本文件** (现状、待办、命令), 规则见第 6 节。

---

## 0. 项目一句话与现状 (2026-10-29)

- **课题**: 医学多模态模型的参数高效增量预训练与指令微调。也是作者 (LOO CHUN QIAN, 导师徐蕾) 的本科毕设。
- **主线模型**: Intern-S2-Preview (35B, 混合专家, 看图 + 文字), LoRA 微调。早期在 Qwen2.5-VL-3B 上做过同一套实验 (Kaggle / 单卡, `configs/`、`notebooks/`), 已结束。
- **主线实验**: 三步训练 ① 文字知识预训练 → ② 问答微调 → ③ 推理训练, 得到模型 A / B / C, 外加换种子 43 重训的 A43 / B43。全部训完、评完。
- **已确立的结论**: 分数主要来自第②步微调; 第①步预训练没有可测提升 (两个 SFT 种子都成立)。第③步推理训练只有单种子结果, 总体收益有限。
- **还没定的**: 成人胸片描述的漏报。换种子就能差约 4 点, 中文对话数据有没有影响要等 CheXbert 复核。
- **报告已同步** (2026-10-25): 英文报告 5.14 已补 A/B 两个 SFT 种子的分数与配对区间, 摘要、分析、局限和复现命令同步更新, PDF 已重出; B43 沿用种子 42 的第①步补丁, 还没有检验预训练本身换种子的影响。
- **周报已同步** (2026-10-25): 第 7 周 PPT 和讲稿已同步报告 v1.6 (37 页), 胸片结果明确标为关键词判断; CheXbert 复核按用户安排放到下周, 执行前仍需队友同意。周报只留本机。
- **人工试用说明已备** (2026-10-29): [docs/MODEL_TRIAL_GUIDE.pdf](docs/MODEL_TRIAL_GUIDE.pdf) 列出补丁位置、合并部署和 7 组双语试题。`gpu_window.sh` 新增 `STEPS=demo` 限时试用, 本机模拟检查通过; 尚未同步或执行服务器试用, 仍需借卡、借盘许可。
- **VS Code 容器连接已修复** (2026-10-29): 用远程 URI 避开 Git Bash 的路径转换, 已确认工作区和新 Bash 终端均在 `/workspace/chunqian`。仅核对图片目录和训练 JSON 存在, 未读取受控内容, 未启动训练或模型服务; 打开方法见 4.10。
- **第 8 周文献材料已备** (2026-10-29): `docs/weekly/week8/` 有 10 篇原始论文的中文精读笔记/PDF、10 页中英 PPT 准备稿和讲稿; 8 篇 PDF 已下载, 另 2 篇提供官方全文链接。区分论文已有贡献、项目已得结论和待验证方向; 个人精读与最终创新方案仍待完成。材料只留本机, 重出命令见 4.11。
- **完整结果**: [results/README.md](results/README.md) (实验日志, 按时间往下记), [docs/report.md](docs/report.md) (英文报告, 第 5 节结果, 第 8 节复现命令)。

---

## 1. 红线 (违反任何一条都可能伤到别人或违反数据协议)

1. **仓库是公开的** (github.com/AugustLoo/MedLoRA)。不写密码、IP、密钥、token、个人联系方式。服务器凭据只在用户本机的 `~/.ssh/config` 里。
2. **CheXpert Plus 是受控数据** (签过 DUA)。
   - 报告原文、图片、逐题记录 (`chexpert_*_preds.jsonl`、`chexbert_*_labels.jsonl`) 只能留在服务器上: 不打印到终端, 不拷回本机, 不发到任何网络服务。
   - 只有汇总文件 `chexpert_<tag>.json`、`chexbert_<tag>.json` 可以带回本机。它们只有比例和计数。
3. **0-3 号卡平时是同学世龙的推理服务在用** (端口 23333, 目录 `/home/ubuntu/Large-Model-Service-Interns2`, 对我们只读)。
   - 要用卡, 必须先由用户征得他同意, 并约好还卡的时间。
   - 停服务一律用 `scripts/server/gpu_window.sh`: 到点自动停、退出时自动恢复。
   - 不要手动 `stop_server.sh` 之后就走开。
4. **只用 0-3 号卡。** 4-7 号是别人的 (6、7 号有 `srb`), 不碰。容器里看到的卡是主机的 6、7 号, 不能用, 容器里只用 CPU。
5. **容器 (user0) 是和队友共用的账号。**
   - tmux 一律加私有 socket: `tmux -L chunqian ...`。
   - 不要 `pkill -f`、`killall`, 只按 PID 杀自己的进程。
   - 不改 `~/.bashrc`, 不 `conda init`。
   - 容器 CPU 也是共用的: 跑 CheXbert 这类长时间 CPU 任务之前, 先让用户问队友。
6. **删服务器上的任何文件之前先问用户。** 没有回收站。例外: 本仓库脚本自己产生、按设计用完即删的合并模型 (`merged/<tag>`)。
7. **主机系统盘只剩约 45 GB。** 合并一个 35B 模型要约 75 GB。
   - 临时借用 `/home/share` 要先由用户问管理员 (苏学长)。
   - 2026-10-24 同意过一次, 用完已还原, 每次都要重新问。
8. **周报 (`docs/weekly/`) 只留本机**, 已在 `.gitignore` 里, 不要强行 `git add -f`。
9. **`results/` 里已有的快照不改。** 新结果写成新文件 `<名字>_<日期>.json`, 旧文件是报告里数字的出处。

---

## 2. 三台机器

| 机器 | 怎么进 | 我们的目录 | 用途 |
|---|---|---|---|
| 本机 (Windows 11, RTX 3050 4 GB) | — | `C:\Users\Y4\fyp\Topic 6\MedLoRA` (本仓库, git) | 写代码、汇总结果、出报告 / PPT。**不能训练 35B** |
| 5090 主机 (8 × RTX 5090 32 GB, 账号 `ubuntu`) | `ssh 5090` | `/home/ubuntu/chunqian/` | 35B 训练、合并、起服务、评测 |
| user0 容器 (同一台机器里的 Docker) | `ssh gpu-user0` | `/workspace/chunqian/` | CheXpert Plus 数据只在这里; CheXbert (CPU) |

- `ssh 5090` 和 `ssh gpu-user0` 是用户本机 `~/.ssh/config` 里配好的别名。
  - 两个都要密码: `5090` 直接要, `gpu-user0` 经过的跳板那一步要。
  - 用户本机已配好无人值守登录 (仓库外的凭据文件 + `SSH_ASKPASS`), 用法写在本机的 `AGENTS.local.md`「登录服务器」一节。
  - 不要把密码写进仓库, 也不要打印出来。
  - 连不上就停下, 请用户处理, 不要反复重试 (多次失败可能被服务器封)。
- **主机和容器上的代码都不是 git 仓库**, 是从本机打包传上去的副本。改了代码要同步过去 (第 4 节)。
- 主机时钟是北京时间, **容器时钟是 UTC**。
- 两边互相访问: 容器 → 主机服务 `http://172.17.0.1:23334`; 主机 → 容器 `http://172.20.0.2:38531` (只用来放「完成」标记)。

### 主机 `/home/ubuntu/chunqian/` 里有什么

```
MedLoRA/            代码副本 (与本仓库同结构)
  outputs/<tag>/    LoRA 补丁 (adapter_model.safetensors) + train_log.jsonl + 检查点; 永久保留
  outputs/eval/     评测结果: <测试>_<tag>.json (汇总) + <测试>_<tag>_preds.jsonl (逐题)
envs/s2train/       训练环境 (conda, torch 2.10+cu128, transformers 5.x); 激活: conda activate /home/ubuntu/chunqian/envs/s2train
data/processed/     训练用 json (slake_train / pubmedqa_sft_train_* / sft35_zh_dialog / sft35_reasoning / cpt35_stage1 …)
data/SLAKE  data/sft6_images  data/medmnist  data/cpt_corpora  data/sft35_raw  data/chexpert_s2
hf/                 HF_HOME (评测数据集缓存; 服务器连不上 huggingface.co, 一律走 HF_ENDPOINT=https://hf-mirror.com)
merged/             临时合并的完整模型 (每个约 75 GB, 评完就删); 平时应为空
logs/               各脚本的日志 (sft35.log、gpu_window.log、<tag>.train.log、<tag>.eval.log、serve_23334.log …)
scripts/            旧的一次性接力脚本 (已整理进仓库 scripts/server/, 新任务用仓库里的)
```

基座模型 (只读): `/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview`。
交给世龙部署的训练版 (回放 300): 同目录 `Intern-S2-Preview-MedLoRA`。

### 容器 `/workspace/chunqian/` 里有什么

- `MedLoRA/`: 代码副本。conda 环境叫 `chunqian`:
  `source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian`
- `MedLoRA/outputs/eval/`: `chexpert_*`、`chexbert_*` 结果。逐题文件不许带出, 见红线 2。
- `data/`: SLAKE 等。CheXpert Plus 在队友目录, 只读, 缩图缓存在 `data/chexpert_test_1024/`。
- `xfer3/`: 给主机看的「完成」标记。

---

## 3. 仓库地图

```
AGENTS.md / README.md
train/interns2/       35B 的全部流程 (每个脚本开头的注释就是它的用法, 先读注释)
  train_lora.py         LoRA 训练 (自写循环, 支持 --resume / --init-adapter / --smoke-steps / --path-map)
  merge_lora.py         补丁合并成完整模型
  serve.sh              用世龙环境里的 LMDeploy 起我们的模型 (端口 23334, tp 4)
  run_sft35.sh          ★ 三步训练第②③步 + 评测: data2|data3|smoke|train|merge|eval|evalzh|status, 模型 a|b|c|cpt1|a43|b43
  run_cpt35.sh          第①步知识预训练: data|smoke|sweep|train|status
  run_6ds.sh / run_cxp.sh   S2-6datasets / S2-CheXpertPlus 两个对照模型
  ablation.sh / external.sh / overnight.sh / serve_for_container.sh / compare_serve.sh   早期 35B 实验与工具
scripts/server/       ★ 服务器上直接跑的运维脚本
  gpu_window.sh         借卡时间窗: 训练 / 评测 / 人工试用 (STEPS=demo), 到点或结束都恢复服务 (主机)
  container_cxp_watch.sh   与上面配对: 容器里自动跑 CheXpert Plus 评测 (容器)
  dl_cpt.sh / dl_omt.sh / omt_files.txt / dl_sft35.sh   三步训练的原始语料下载 (主机, 走 hf-mirror)
eval/                 各测试的评测脚本 (都经 OpenAI 兼容接口调用模型: MEDVLM_API_BASE)
  eval_slake / eval_medvqa (VQA-RAD, PathVQA) / eval_medqa / eval_cmexam / eval_pneumonia / eval_pubmedqa / eval_general (TextVQA) / eval_mmbench
  eval_chexpert.py      成人胸片漏报 (容器)      chexbert_judge.py   用 CheXbert 重判描述 (容器 CPU)
  eval_openended.py     自由描述探针
data/                 下载与转换脚本 (convert_*.py); data/raw、data/processed 不进 git; pubmedqa_split.json 是固定的考卷划分
medvlm/               共用代码: 提示词 prompts.py (训练和评测必须用同一套)、指标 metrics.py、远程调用 remote.py
scripts/              本机脚本: 汇总 (summarize_3stage.py、summarize_seed43.py)、配对区间 (bootstrap_ci.py)、
                      报告 PDF (make_report_pdf.py)、试用说明 PDF (make_model_trial_pdf.py)、文献笔记 PDF (make_literature_pdf.py)、
                      图 (make_fig*.py)、周报 (make_weekN_ppt.py, 包括第 8 周文献准备稿)
results/              结果快照 json + README.md (实验日志); 报告里的每个数字都能在这里找到出处
docs/                 report.md/.pdf (英文报告)、cards/ (模型卡、数据卡)、PLAN_35B_CPT.md (三步训练方案)、
                      INTERNS2.md (35B 操作记录)、TRAINING_35B.md (35B 训练说明, 中文)、SERVER.md (早期 3B 的服务器记录)、
                      COMPARE_API.md / API.md (给同学调用的接口)、refs.bib
tests/                pytest, 只测数据转换和训练脚本的纯逻辑, 本机就能跑
configs/ notebooks/   早期 3B 实验 (LLaMA-Factory yaml、Kaggle notebook), 不再改
```

模型标签 (tag) 是贯穿全项目的名字: 补丁在 `outputs/<tag>/`, 评测结果是 `outputs/eval/<测试>_<tag>.json`。

| 简称 | tag | 是什么 |
|---|---|---|
| 原版 | `interns2_base` | 不训练的基座 |
| S2-2datasets | `interns2_mix_300` | SLAKE + 300 条 PubMedQA 回放 (交给世龙部署的那个) |
| S2-6datasets | `interns2_6ds` | 六个测试集的训练部分 |
| S2-CheXpertPlus | `interns2_cxp` | 只用 1 万对胸片 → 报告 |
| cpt1 | `interns2_cpt1` | 只做第①步 |
| A / B / C | `interns2_s2a` / `interns2_s2b` / `interns2_s3c` | 三步训练 (种子 42) |
| A43 / B43 | `interns2_s2a_s43` / `interns2_s2b_s43` | A、B 换种子 43 |

---

## 4. 常用操作

### 4.1 本机检查 (改完代码先跑)

```bash
cd "/c/Users/Y4/fyp/Topic 6/MedLoRA"
python -m pytest -q tests                  # 本机检查, 现在 51 passed 1 skipped
bash -n train/interns2/*.sh scripts/server/*.sh
```

Windows 注意事项:
- 打印中文前设 `PYTHONIOENCODING=utf-8`, 否则 GBK 报错。
- Git Bash 的 heredoc 会吞反斜杠: 含反斜杠的脚本先写成文件再运行。
- shell 脚本必须是 LF 换行 (`.gitattributes` 已规定)。

### 4.2 把改过的代码同步到服务器

服务器上不是 git。只打包改过的文件:

```bash
# 本机 Git Bash, 仓库根目录
tar -czf 5090-update.tar.gz train/interns2/run_sft35.sh scripts/server/gpu_window.sh     # 列出改过的文件
scp 5090-update.tar.gz 5090:/home/ubuntu/chunqian/
ssh 5090 'cd /home/ubuntu/chunqian/MedLoRA && tar -xzf ../5090-update.tar.gz --keep-directory-symlink && md5sum train/interns2/run_sft35.sh'
md5sum train/interns2/run_sft35.sh                                                        # 两边一致才算同步好
```

- **必须带 `--keep-directory-symlink`**: 容器上的 `data/raw` 是指向 `/workspace/chunqian/data` 的符号链接, 不带会被换成空目录 (主机上是普通目录, 带上也无害)。
- **正在运行的 bash 脚本不能原地覆盖。** bash 边跑边读文件, 覆盖会让正在跑的流程读到错位的内容。
  先 `ps -eo pid,cmd | grep <脚本名>` 看有没有在跑。在跑就等它结束, 或者先写到临时文件再 `mv` 过去 (换新文件, 正在跑的进程继续读旧的)。
- 容器同理: `scp` 到 `gpu-user0:/workspace/chunqian/`, 在 `/workspace/chunqian/MedLoRA` 里解压。

### 4.3 看服务器状态 (只读, 随时可以做)

```bash
ssh 5090 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; ss -ltn | grep -E ":2333[34] "; df -h / /home/share; tmux ls'
ssh 5090 'tail -20 /home/ubuntu/chunqian/logs/gpu_window.log'
ssh 5090 'tail -1 /home/ubuntu/chunqian/MedLoRA/outputs/<tag>/train_log.jsonl'     # 训练进度, 有 eta_min
ssh 5090 'cd /home/ubuntu/chunqian/MedLoRA && bash train/interns2/run_sft35.sh status'
```

- 23333 在监听 = 世龙的服务开着; 23334 = 我们的模型服务。
- 0-3 号卡显存 > 1 GB 就是有人在用。

### 4.4 训练 / 评测一个 35B 模型 (要借卡)

前提: 用户已征得世龙同意并约好截止时间; 要评测的话, 合并空间已解决 (4.5)。

```bash
# 主机, 先同步好代码 (4.2)
ssh 5090
tmux new-session -d -s gpuwin "DEADLINE_AT='2026-11-01 14:00' MODELS='a43 b43' STEPS=train,eval \
    bash /home/ubuntu/chunqian/MedLoRA/scripts/server/gpu_window.sh"
```

- 开头先看 2 分钟: 世龙的服务有请求或卡在忙就不开始, 直接退出。
- 然后停他的服务, 按顺序训练 / 评测; 结束、失败、被中断、到截止时间, 都会恢复他的服务。
- 训练每 100 步存断点。重跑同一命令会自动接着训; 已有补丁的跳过训练, 已有结果的测试跳过评测。
- 时长 (0-3 号卡):
  - 第②步一个模型 5,241 步 × 约 6.7 秒 ≈ 10 小时。
  - 合并约 3 分钟。
  - 主机评测约 2 小时: 四张表、SLAKE 中文、CMExam、去掉图像、VQA-RAD、PathVQA、MedQA、PneumoniaMNIST。
- 新的训练变体: 先在 `run_sft35.sh` 的 `model_of` 里加一行, 同时在 `gpu_window.sh` 的 `tag_of` 里加同一个映射。

要同时跑成人胸片评测 (CheXpert Plus, 约 1 小时, 和主机评测并行), 先在容器里起配对脚本, 把它打印的 `CXP_SIG=...` 传给主机:

```bash
ssh gpu-user0
TAGS="interns2_s2a_s43 interns2_s2b_s43" END_AT="2026-11-01 14:10" \
  tmux -L chunqian new-session -d -s cxpwatch "bash /workspace/chunqian/MedLoRA/scripts/server/container_cxp_watch.sh"
head -1 /workspace/chunqian/MedLoRA/outputs/eval/cxp_watch.log        # CXP_SIG=http://172.20.0.2:38531/<随机串>
# 回到主机: 在 gpu_window.sh 的命令里加 CXP_SIG='<上面那个地址>'
```

### 4.5 硬盘不够合并时 (要管理员同意)

`run_sft35.sh merge` 会检查合并目录所在盘的剩余空间, 少于 80 GB 就拒绝。系统盘不够时, 经苏学长同意后把合并目录临时指到 `/home/share`:

```bash
# 主机; merged/ 必须是空的才做
R=/home/ubuntu/chunqian; ls -A $R/merged
mkdir -p /home/share/chunqian_tmp && chmod 700 /home/share/chunqian_tmp
rmdir $R/merged && ln -s /home/share/chunqian_tmp $R/merged
# 用完 (gpu_window.sh 每评完一个就删一个合并模型) 还原:
rmdir /home/share/chunqian_tmp && rm $R/merged && mkdir $R/merged
```

### 4.6 CheXbert 复核 (容器 CPU, 先问队友)

`eval_chexpert.py` 用关键词判断描述里有没有报异常。`chexbert_judge.py` 用 CheXbert 模型重读同一批描述, 结论以它为准。

```bash
ssh gpu-user0
tmux -L chunqian new-session -d -s chexbert "source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian \
  && cd /workspace/chunqian/MedLoRA && PYTHONIOENCODING=utf-8 python eval/chexbert_judge.py score interns2_s2a_s43 interns2_s2b_s43 \
  >> outputs/eval/chexbert_night.log 2>&1"
tail -5 /workspace/chunqian/MedLoRA/outputs/eval/chexbert_night.log
```

(conda 激活要写在 tmux 命令里面: tmux 的新会话不继承当前终端的环境。)

- 需要该模型的 `chexpert_<tag>_preds.jsonl` 已经在容器里。
- 输出: `chexbert_<tag>.json` 是汇总, 可带回本机; `chexbert_<tag>_labels.jsonl` 是逐条, 留在服务器。
- 尺子校验 (`validate --limit 300`) 已做过, 一致率 96.2%, 不用重做。

### 4.7 结果带回本机并汇总

```bash
# 主机上的汇总和逐题记录 (逐题记录可以带回, 配对区间要用; 但 chexpert / chexbert 的逐题文件不行)
ssh 5090 'cd /home/ubuntu/chunqian/MedLoRA && tar czf - outputs/eval/*_<tag>.json outputs/eval/*_<tag>_preds.jsonl outputs/<tag>/train_log.jsonl' | tar xzf -
# 容器上只拿汇总
scp gpu-user0:/workspace/chunqian/MedLoRA/outputs/eval/chexpert_<tag>.json outputs/eval/

python scripts/bootstrap_ci.py <对照tag> <tag1> <tag2> --out results/<名字>_<日期>.json    # 配对 95% 区间, 区间不跨 0 才算有差别
python scripts/summarize_seed43.py      # 或 summarize_3stage.py; 新实验照它们的样子写一个 summarize_*.py
```

然后:
1. 在 `results/README.md` 末尾加一节: 表格和「读法」。
2. 需要的话更新 `docs/report.md` (第 5 节结果、第 6 节分析、第 7 节局限、第 8 节复现命令, 以及摘要)。
3. 运行 `python scripts/make_report_pdf.py` 重出 `docs/report.pdf`。要本机装有 Edge; 输出路径用绝对路径。

`outputs/` 不进 git, 本机只放带回来的副本。

### 4.8 下载原始语料 / 重做数据 (主机, 不占卡)

```bash
bash scripts/server/dl_cpt.sh        # 第①步语料: OpenMedText 教材 (失败时用 dl_omt.sh 逐个文件下)、临床指南、MMedC 中文
bash scripts/server/dl_sft35.sh      # 第②③步: HuatuoGPT、DISC-Med、CMExam、MedMCQA、medical-o1
bash train/interns2/run_cpt35.sh data && bash train/interns2/run_sft35.sh data2 && bash train/interns2/run_sft35.sh data3
```

数据来源、许可、清洗规则见 `docs/PLAN_35B_CPT.md` 和 `docs/cards/DATA_CARD.md`。

### 4.9 人工提问试用 (主机, 要借卡)

按 `docs/MODEL_TRIAL_GUIDE.pdf` 操作: 先获准借卡、约定截止时间, 需要临时盘时重新获准借盘, 再同步新版 `gpu_window.sh`。`STEPS=demo`、`MODELS=a` 表示只合并 A42 并启动供人工提问, 不训练、不跑整套评测; MODELS 一次只能指定一个。合并目录需至少 80 GB 空间, 距截止至少 15 分钟, 不覆盖已有同名模型或已有的 23334 服务。结束时删除本次生成的临时合并模型、保留补丁并恢复原服务。该模式尚未在服务器实跑。

本机重出说明 (输出用绝对路径):

```bash
python scripts/make_model_trial_pdf.py 'C:/Users/Y4/fyp/Topic 6/MedLoRA/docs/MODEL_TRIAL_GUIDE.pdf'
```

### 4.10 用本机 VS Code 打开容器目录 (不占卡)

本机 Git Bash 中执行, 先按 `AGENTS.local.md` 准备登录环境, 或在 VS Code 提示时完成登录:

```bash
MSYS_NO_PATHCONV=1 code --reuse-window --folder-uri 'vscode-remote://ssh-remote+gpu-user0/workspace/chunqian'
```

使用远程 URI 保留 Linux 路径。直接把 `/workspace/chunqian` 作为普通参数传给 Windows 的 code, Git Bash 会将其转换成错误的本机路径, 即使 SSH 成功也无法打开目录。修复后检查左侧资源管理器, 再新建终端; 实际工作目录应为 `/workspace/chunqian`, 不能只看 SSH 连接成功的日志。受控图片和报告仍按第 1 节限制处理。

### 4.11 第 8 周文献材料 (本机)

材料在 `docs/weekly/week8/`, 依第 1 节只留本机。先按实际精读情况修改笔记/讲稿和生成脚本里的状态, 不把准备材料当作个人已读记录, 不把候选方向写成已验证创新。重出:

```bash
python scripts/make_literature_pdf.py 'C:/Users/Y4/fyp/Topic 6/MedLoRA/docs/weekly/week8/literature_review.md' 'C:/Users/Y4/fyp/Topic 6/MedLoRA/docs/weekly/week8/literature_review.pdf'
python scripts/make_week8_ppt.py 'C:/Users/Y4/fyp/Topic 6/MedLoRA/docs/weekly/Weekly_Report_Template.pptx' 'C:/Users/Y4/fyp/Topic 6/MedLoRA/docs/weekly/week8/Topic6_WeeklyReport8_Literature_Draft.pptx'
```

PDF 用 Edge 排版; PPT 生成后按本机规则用 PowerPoint 导出各页 PNG 检查。论文全文仅用于阅读, 不涉及服务器受控数据或新增训练。

---

## 5. 写代码的约定

- **注释和日志用中文, 大白话**, 与现有代码一致。代码标识符、提交信息用英文。
- **服务器脚本要能重跑**: 做完的步骤自动跳过; 长任务有断点; 失败时日志里写清楚是哪一步 (日志行以 `!!` 开头表示出错)。
- 要用卡的脚本自己检查 23333 和 0-3 号卡 (照 `run_sft35.sh` 的 `need_gpus`), **从不自己停世龙的服务**。停服务只在 `gpu_window.sh` 里做。
- 训练和评测用同一套提示词 (`medvlm/prompts.py`), 改一边必须改两边。
- 评测一律经接口 (`MEDVLM_API_BASE`, 关思考: `MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'`)。
- 测试集与训练数据隔离: 转换脚本里有按题目去重 / 排除测试集的逻辑, 新数据也要做。PubMedQA 只用 `data/pubmedqa_split.json` 的 train 半边训练。
- 结论只写两个种子都成立、且配对区间不跨 0 的; 单种子的结果要标「单种子」。
- 提交前: `pytest`, `bash -n`, 再扫一遍没有密码 / IP / 受控数据。改动小而集中, 一次提交一件事。

---

## 6. 维护本文件 (每次做完都要做)

每完成一件事 (训练、评测、新脚本、改流程、更新报告等), 收尾时更新本文件, 和改动一起提交推送:

1. 第 0 节「现状」: 改日期, 写进新结论。
2. 第 7 节「待办」: 做完的删掉, 新冒出来的加上, 写清楚卡在谁那里。
3. 命令、路径、脚本有变化, 同步改第 3、4 节。写进去的命令要先核对路径确实存在。
4. 个人事务 (联系人、毕设表格、周会上的决定) 改本机的 `AGENTS.local.md`, 不进 git。
5. 提交前照第 1 节检查一遍: 不写密码、IP、受控数据。

## 7. 待办 (2026-10-29)

0. **导师 10-29 给了新的论文规划** (目标 CCF A 类会议, 方向是「参数高效适配中的视觉证据保持」)。本机执行方案在 `docs/private/` (不进 git, 投稿前不公开), **开工前先读它**; 下面各项的先后以它为准。第 1–2 周的重点: 多种子核查描述漏报是否稳定、简单回放能否解决。

1. **CheXbert 复核 (安排到下周)**: `interns2_s2a`、`interns2_s2b`、`interns2_s3c`、`interns2_cpt1`、`interns2_s2a_s43`、`interns2_s2b_s43` 还没做 (4.6)。先做 A/B 两个种子, 再做 C 和 cpt1; 执行前等用户征得队友同意并约定可用时段。
2. **要不要加种子 44**: 10-29 周会上请导师决定。要做的话, 先在 `run_sft35.sh` 里加 `a44`, 用 `gpu_window.sh` 跑, 约 12 小时卡。
3. **要不要把 CheXpert Plus 胸片报告混进问答微调 (A + 报告)**: 10-29 周会上请导师决定, 第 7 周 PPT 第 10 页第 ② 问。
   - 理由: 只练报告的 `interns2_cxp` 胸片最平衡 (CheXbert 描述漏报 10.2%, 是非题误报 25.1%), 但 SLAKE 是非题掉回 83.9 (A 是 94.2); 混训是想两边都保住, 不保证更好。
   - 导师同意后的顺序: 先做完待办 1 的 CheXbert → 再在 `run_sft35.sh` 加一个模型 (第②步数据 + 一部分 B3 候选池报告, 只用训练划分, 与测试 3,189 张不同病人) → 种子 42、43 各训一次 → 看胸片漏报 / 误报 / 矛盾和 SLAKE 等问答。每个种子约 12 小时以上卡, 要先约世龙。
   - 报告数据是受控数据: 只在服务器上生成和使用 (见 `run_cxp.sh` 和 docs/INTERNS2.md「S2-CheXpertPlus」), 不带回本机。
4. 每周四周会前做周报 PPT: `scripts/make_weekN_ppt.py`, 模板 `docs/weekly/Weekly_Report_Template.pptx`, 输出在 `docs/weekly/weekN/`, 只留本机。第 7 周 PPT / 讲稿已同步 (10-27 加了 CheXpert Plus 混训的决策问题); 周会后的决定另行记录。生成脚本可带第二个参数指定输出路径 (PPT 被 WPS 打开时先输出到别处预览)。
5. **A 模型人工试用**: 说明 PDF 和限时 demo 模式已备; 等用户取得借卡时段, 需要临时盘时另获借盘许可。届时先同步代码、核对补丁和空间, 再启动服务, 用固定题目保存实际回答。已用 VS Code 连接容器并核对数据路径存在, 尚未启动模型试用或同步新运维脚本。
6. **10 篇文献精读与创新性汇报 (11-05)**: 准备材料见 4.11; 用户需逐篇核对方法与关键实验、用自己的话复述创新和局限。先比较最近方法, 再与导师确定研究问题、基线和验证方案; 候选选样/混训方向尚未实施, 不预填个人已读或新实验完成。
