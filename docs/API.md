# 调用接口：用远程部署的模型跑四张评估表

2026-09-25 · 用于老师提供的 35B 模型，模型部署在同学的服务器容器里

## 一句话

模型由服务端部署成 **OpenAI 兼容接口**，我们的评估脚本通过 HTTP 发题目、收答案、在本地打分。评估脚本和打分逻辑一行没改，只多了一个远程后端，设一个环境变量就切换。

```
评估脚本 ──(题目 + 图片)──▶ POST /v1/chat/completions ──▶ 35B 模型
        ◀──────(答案文本)────────────────────────────────
```

代码在 `medvlm/remote.py`，切换逻辑在 `medvlm/model.py`。

---

## 一、需要服务端（同学）提供的

| 项目 | 要求 | 为什么 |
|---|---|---|
| 接口 | `GET /v1/models` 和 `POST /v1/chat/completions`，OpenAI 格式 | vLLM、SGLang、LMDeploy 默认就是这个格式 |
| 图片 | 接受 `image_url` 字段里的 base64 data URL（`data:image/png;base64,...`） | 三张表要看图 |
| 解码 | 尊重请求里的 `temperature: 0` | 和 3B 的贪心解码一致，结果可复现 |
| 模型名 | 告诉我们 `/v1/models` 里的模型名 | 评估时要填 |
| 地址 | 服务的 IP 和端口；是否需要 API key | 填进环境变量 |

用 vLLM 部署时大致是下面这样，具体参数以模型为准，由同学决定：

```bash
vllm serve /path/to/35b-model \
  --served-model-name big35b \
  --tensor-parallel-size 8 \
  --port 8000 \
  --limit-mm-per-prompt image=1
```

**要问清楚的四件事**

1. 这个 35B 能不能看图？如果是纯文字模型，只能跑 PubMedQA 一张表。
2. 它是不是推理模型（回答前会先输出一段 `<think>` 思考）？是的话需要关闭思考模式，见第四节。
3. 服务地址、端口、模型名、要不要 API key。
4. 容器里能不能放评估数据（见第三节），还是评估脚本要在别的机器上跑、通过网络调用。

---

## 二、我们这边怎么跑

在能访问服务的机器上（同一个容器里就用 `127.0.0.1`）：

```bash
cd MedLoRA
export MEDVLM_API_BASE=http://127.0.0.1:8000/v1
# export MEDVLM_API_KEY=...          # 服务端要求鉴权时才设，不要写进代码或 git

# 第一步：连通性检查，一分钟
python scripts/check_api.py

# 第二步：先跑几题看看
python eval/eval_slake.py --model big35b --tag big35b_base --limit 20

# 第三步：四张表全跑
MODEL=big35b bash train/eval_all.sh big35b_base
```

结果和以前一样写进 `outputs/eval/*_big35b_base.json`，汇总 JSON 里多一个 `backend` 字段，记录这是远程接口跑出来的、服务地址和模型名，之后不会和本地结果混淆。

`check_api.py` 检查四件事：连得上、纯文字题能答、看图题能答、回答是不是只有一个词。看图题用的是脚本自己画的一张红色方块图，不需要任何数据集。

---

## 三、评估数据要放在跑评估脚本的机器上

图片是评估脚本读出来再发给服务的，所以**数据要在评估脚本那一侧，不在模型那一侧**。

| 表 | 数据 | 来源 |
|---|---|---|
| SLAKE | `data/raw/SLAKE/`（题目 + 图片） | 从学院服务器 `/workspace/chunqian/MedLoRA/data/raw/SLAKE` 拷贝 |
| PubMedQA | HuggingFace `qiaojin/PubMedQA` | 联网下载，或拷贝 HF 缓存 |
| TextVQA | HuggingFace `lmms-lab/textvqa` | 同上 |
| MMBench | HuggingFace `lmms-lab/MMBench` | 同上 |

容器不能联网时，把学院服务器上的 `~/.cache/huggingface/` 里对应的三个数据集目录拷过去，再设 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。

远程模式不需要装 torch，只要 `pillow`、`datasets`、`scikit-learn`、`tqdm`。

---

## 四、和 3B 实验保持一致的地方

| | 3B 本地 | 35B 远程 | 做法 |
|---|---|---|---|
| 解码 | `do_sample=False` | `temperature=0, top_p=1` | 请求里固定 |
| 最长回答 | 各表不同（8 / 16 / 32 个 token） | 相同 | 评估脚本原样传入 `max_tokens` |
| 图片大小 | processor 的 `max_pixels` | 客户端先按同一个 `max_pixels` 等比缩小 | `encode_image` |
| 提示词 | `medvlm/prompts.py` | 同一份 | 没改 |
| 抽题 | 固定种子 42 | 相同 | 没改 |

聊天模板由服务端按模型自己的模板套，这一点两边不可能完全一样，报告里要写明。

**推理模型**：如果 35B 回答前会先输出 `<think>` 思考段，8 个 token 的上限会把回答截断。客户端会自动去掉完整的思考段，被截断的记为答错。更好的办法是关闭思考模式：

```bash
export MEDVLM_API_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'
```

这个变量里的 JSON 会原样并入每个请求体，其他服务端参数也可以这样传。

---

## 五、adapter 怎么办

远程模式不能直接加载本地的 LoRA adapter 路径。要评估微调后的 35B，让服务端把 adapter 挂成一个模型名，再用这个名字调用：

```bash
# 服务端
vllm serve /path/to/35b-model --enable-lora --lora-modules big35b_sft300=/path/to/adapter ...
# 我们这边
MODEL=big35b_sft300 bash train/eval_all.sh big35b_sft300
```

训练 35B 的 LoRA 不走这个接口，要在有显卡的容器里直接用 LLaMA-Factory 跑，那是另一件事。

---

## 六、自测

没有真模型时，用假服务测试调用代码：

```bash
python scripts/mock_api_server.py --port 8765          # 终端一
export MEDVLM_API_BASE=http://127.0.0.1:8765/v1          # 终端二
python scripts/check_api.py
```

假服务带图片就答 `red`、否则答 `yes`，并检查请求格式（模型名、`temperature=0`、图片是 data URL），格式不对返回 400。2026-09-25 在本机测过：连通性检查四项通过，SLAKE 评估脚本走远程路径跑通，`backend` 字段正确写入；服务没起时报「连不上」，远程模式传 adapter 路径时给出挂载提示。

## 七、安全

- 管理员账号的密码、API key 只放环境变量，不写进仓库任何文件。
- 评估数据里的 SLAKE 测试集和 PubMedQA 考卷半边只用于评估，不拿去训练 35B。
- 在同学的容器里只动自己建的目录。
