# Intern-S2 原版 vs 训练版 · 对比接口说明

给在服务器上做对比界面（UI）的同学。两个模型通过同一个地址提供服务，**一次只开一个**：先开原版问一遍，切换到训练版，再问同样的问题，最后把两边答案放在一起比。

- 负责人：Chunqian（Topic 6 / Task 1.3）
- 更新：2026-10-10（加了第三个模型 S2-6datasets，见第一节末尾）

---

## 一、两个模型是什么

| | 原版 | 训练版 |
|---|---|---|
| 简称（工具和答案文件里用） | `intern-s2-base` | `intern-s2-medlora` |
| 是什么 | 老师提供的 Intern-S2-Preview，没有训练过 | 在原版上做了医学微调，就是交给世龙部署的那个 |
| 服务器目录（只读） | `/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview` | `/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview-MedLoRA` |
| 大小 | 35B 参数，混合专家结构（每次约激活 3B），73 GB | 同左 |

### 训练版用了什么数据训练

只用了两份英文数据，合计 5,219 条：

| 数据 | 条数 | 内容 | 许可 |
|---|---|---|---|
| **SLAKE** 训练集（英文部分） | 4,919 | 医学影像问答：CT、MRI、X 光等图片加短问答，例如「这是什么检查？」「病灶在哪个器官？」 | CC BY 4.0 |
| **PubMedQA**（专家标注集的训练半边） | 300 | 纯文字题：给一段论文摘要，回答 yes / no / maybe，三类各 100 条。用来教模型在证据不足时敢说「不确定」 | MIT |

**没有用来训练的数据**：CheXpert Plus、VQA-RAD、PathVQA、MedQA、PneumoniaMNIST，以及 SLAKE 测试集、PubMedQA 的另一半。这些都**只用来考试**。

### 怎么训练的

- **方法**：LoRA（只训练一小块「补丁」，原模型参数不动）。
  - rank 16、alpha 32、dropout 0.05；
  - 只挂在语言部分的注意力层和共享专家上，可训练参数 1,917 万，占全部参数的 0.054%；
  - 看图部分（视觉编码器）不训练。
- **参数**：学习率 1e-4，3 轮，有效 batch 16，图片上限 512×512 像素，随机种子 42，训练时**关闭思考模式**。
- **耗时**：4 张 RTX 5090，978 步，1 小时 58 分。
- **交付**：训练好的补丁合并进原模型，成为一个完整模型（世龙的推理环境加载不了单独的补丁）。
- 训练代码：`train/interns2/train_lora.py`；合并代码：`train/interns2/merge_lora.py`。

### 第三个模型：S2-6datasets（2026-10-10 起可用）

| | S2-6datasets |
|---|---|
| 简称 | `intern-s2-6ds` |
| 服务器目录（只读） | `/home/ubuntu/chunqian/merged/interns2_6ds` |
| 训练数据 | 上面两份，再加四个数据集的**训练部分**：VQA-RAD 734、PathVQA 5,000、MedQA 3,000、PneumoniaMNIST 2,000，合计 15,953 条 |
| 训练方法 | 和训练版完全一样，只换了数据 |
| 已知表现 | SLAKE、PubMedQA、TextVQA、MMBench 四项和训练版基本一样；病理图题（PathVQA）92.7（训练版 76.8）、医师考试题 85.3、儿童胸片判断肺炎 95.4 且描述时很少再把肺炎说成正常（漏报 20.8%，训练版 77.7%）。这四项它练过同类题（但没练过测试题）。成人胸片的描述还没测 |

切换：`bash train/interns2/compare_serve.sh switch six`。上面「训练版」指的仍是交给世龙的那个（S2-2datasets）。

---

## 二、已知表现（先看这个，对比时心里有数）

以下全部是**关闭思考模式**时的成绩，训练版就是上面这一个模型（种子 42），每项只测过一次。

| 测试 | 原版 | 训练版 | 说明 |
|---|---|---|---|
| SLAKE 是非题 | 84.4 | **94.5** | 主任务，训练用的同一套题的测试部分 |
| SLAKE 开放题（完全答对） | 67.1 | **86.4** | |
| PubMedQA macro-F1 | 61.8 | **67.3** | 「不确定」答对：7 → 22（共 55 题） |
| VQA-RAD 是非题 | 80.5 | **85.3** | 另一套放射科题，没训练过，说明不是背题 |
| TextVQA（看图读字） | 88.8 | 86.7 | 略降 |
| MMBench（通用看图选择题） | 93.6 | 94.0 | 基本不变 |
| PathVQA 是非题（病理切片） | 80.1 | 76.8 | 下降，而且变得爱答「no」 |
| MedQA（医师考试文字题） | 85.0 | 83.6 | 小降 |
| **胸片描述：有病的片子被写成「正常」** | 29.2% | **68.5%** ⚠️ | CheXpert Plus 2,733 张有病的成人胸片 |

**一句话**：
- 训练版**医学短答明显更好**，也更敢说「不确定」；
- 但**让它描述胸片时，很容易把有病写成正常**，这一点比原版差很多。

对比时，挑几道「描述这张胸片」的题，能看到这个差别。

> ⚠️ 两个模型都**只能用于研究和演示，不能用于临床判断**。

---

## 三、使用流程

```
通知世龙 → 开原版 → 问一批题 → 切到训练版 → 问同一批题 → 合并 → 关服务 → 告诉世龙
```

服务器上（ubuntu 账号）：

```bash
cd /home/ubuntu/chunqian/MedLoRA

# 0. 先通知世龙：要用 0-3 号卡，请他停掉他的服务（端口 23333）。
#    他的服务开着时，下面的脚本会拒绝启动，不会去停他的服务。

# 1. 开原版（约 3-6 分钟）
bash train/interns2/compare_serve.sh switch base

# 2. 提问（单问，或批量跑一个问题文件）
conda activate /home/ubuntu/chunqian/envs/s2train
python scripts/compare_ask.py ask "What modality is used in this image?" --image /path/to/image.png
python scripts/compare_ask.py batch questions.jsonl

# 3. 切到训练版，用同一个问题文件再跑一遍
bash train/interns2/compare_serve.sh switch final
python scripts/compare_ask.py batch questions.jsonl
#    （想再比 S2-6datasets：switch six，再 batch 一遍；merge 时把三个答案文件都列出来）

# 4. 合并两边答案（按题目对齐）
python scripts/compare_ask.py merge
#   → outputs/compare/compare.jsonl
#     每行: {"id", "question", "image", "intern-s2-base": "...", "intern-s2-medlora": "..."}

# 5. 用完关掉，告诉世龙可以开回他的服务
bash train/interns2/compare_serve.sh stop
```

随时查看状态：`bash train/interns2/compare_serve.sh status`

**问题文件** `questions.jsonl` 每行一道题，`id` 不能重复，`image` 可以不写：

```json
{"id": "q1", "question": "Is there any abnormality in this chest X-ray? Answer yes or no."}
{"id": "q2", "question": "Describe this chest X-ray in two or three sentences, including any abnormal findings.", "image": "/home/ubuntu/xxx/chest.png"}
```

- 批量中途断了，重跑同一条命令会**接着跑**，已答过的题不重问。
- 答案存在 `outputs/compare/answers_<模型名>.jsonl`。

---

## 四、给 UI 直接调用的接口

UI 也可以不经过上面的小工具，直接调接口。接口是 **OpenAI 兼容格式**，由 LMDeploy 提供。

| 项目 | 值 |
|---|---|
| 地址（在主机上） | `http://127.0.0.1:23334/v1` |
| 地址（在 user0 等容器里） | `http://172.17.0.1:23334/v1` |
| 查当前是哪个模型 | `GET /v1/models`，返回的 `data[0].id` 是**模型目录路径**：结尾是 `Intern-S2-Preview` 就是原版，结尾是 `Intern-S2-Preview-MedLoRA` 就是训练版，结尾是 `interns2_6ds` 就是 S2-6datasets |
| 提问 | `POST /v1/chat/completions` |
| 鉴权 | 不需要 |

### 必须带的设置

1. **关闭思考模式**：请求里加 `"chat_template_kwargs": {"enable_thinking": false}`。
   - 不关的话，模型会先写一大段推理，慢很多，而且和我们测出的成绩不是同一种设置。
2. **`temperature: 0`**：同一道题每次答案基本不变，两边对比才公平。
3. **`model` 原样填 `/v1/models` 返回的 `id`**（一个目录路径）。UI 按路径结尾显示成「原版」或「训练版」。

> 为什么不给服务起个短名字：服务的启动参数和我们测成绩时保持完全一致，一个都不改，这样对比看到的回答就是测出来的那个模型的真实表现。

### 文字题（curl）

```bash
# 原版时 model 填下面这个路径; 训练版时换成 .../Intern-S2-Preview-MedLoRA (以 /v1/models 返回的为准)
curl -s http://127.0.0.1:23334/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview",
  "messages": [{"role": "user", "content": "What is pneumothorax? Answer in one sentence."}],
  "temperature": 0, "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

### 看图题（Python，需要 requests 和 Pillow）

图片以 base64 data URL 发送。建议和我们测试时一样，先等比缩到不超过 512×28×28 像素（约 40 万像素）：

```python
import base64, io, math, requests
from PIL import Image

BASE = "http://127.0.0.1:23334/v1"
model = requests.get(f"{BASE}/models", timeout=10).json()["data"][0]["id"]

img = Image.open("chest.png").convert("RGB")
w, h = img.size
limit = 512 * 28 * 28
if w * h > limit:
    s = math.sqrt(limit / (w * h))
    img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

r = requests.post(f"{BASE}/chat/completions", timeout=600, json={
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Describe this chest X-ray in two or three sentences."}]}],
    "temperature": 0, "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": False},
}).json()
print(model, r["choices"][0]["message"]["content"])
```

### UI 要处理的几种情况

| 情况 | 现象 | 建议 UI 显示 |
|---|---|---|
| 正在切换模型 | 连接被拒绝，或 `/v1/models` 超时 | 「模型切换中，请稍等几分钟」 |
| 两个模型都没开 | 同上 | 提示运行 `compare_serve.sh status` 查看 |
| 回答被截断 | `finish_reason` 为 `"length"` | 标注「回答过长被截断」，可以调大 `max_tokens` |
| 回答里出现 `<think>…</think>` | 忘了关思考模式 | 检查请求有没有带 `enable_thinking: false` |

- 每次提问都**先查 `/v1/models`，按返回的路径记下是哪个模型**，再存答案。不要靠「我以为现在开的是哪个」，免得两边搞混。
- 速度参考：短答几秒，描述类一二十秒，图片大、回答长会更久。超时建议设 600 秒。

---

## 五、规矩（一定要遵守）

1. **一次只开一个模型**，用完就 `stop`。0-3 号卡是和世龙共用的，**6、7 号卡不能用**。
2. **不要自己去停世龙的服务**，先找他本人。切换脚本也不会替你停。
3. **两个模型目录只读**：不改、不复制、不移动。
4. **演示只用公开图片**，例如 SLAKE、VQA-RAD 的图片，或网上公开的教学图片。**不要用 CheXpert Plus 的病人片子**，那是受控数据，提问记录和截图也不能带出服务器。
5. 不要把接口开放到服务器外面。

---

## 六、相关文件

| 文件 | 作用 |
|---|---|
| `train/interns2/compare_serve.sh` | 切换 / 查看 / 关闭对比服务 |
| `train/interns2/serve.sh` | 起 LMDeploy 服务（切换脚本内部调用） |
| `scripts/compare_ask.py` | 单问、批量提问、合并答案 |
| `scripts/mock_api_server.py` | 假接口，不占显卡，用来离线测试 UI：`python scripts/mock_api_server.py --port 8765 --model-name /x/models/Intern-S2-Preview`（假装是原版；换成 `…/Intern-S2-Preview-MedLoRA` 就是假装训练版） |
| `docs/cards/MODEL_CARD.md` | 模型卡（训练细节、全部成绩） |
| `docs/cards/DATA_CARD.md` | 数据卡（每个数据集的来源和许可） |
