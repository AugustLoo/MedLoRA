"""生成模型补丁、限时部署和双语试题说明 PDF; 仅在本机排版, 不连接服务器。

用法: python scripts/make_model_trial_pdf.py <输出 PDF 的绝对路径>
"""
from pathlib import Path
import subprocess
import sys

import mistune


REPO = Path(__file__).resolve().parents[1]
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
PDF = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "docs/MODEL_TRIAL_GUIDE.pdf"
HTML = REPO / "outputs/model_trial_guide.html"

PAGES = [
    ("模型与补丁位置", "Models and adapter locations", r"""
**用途：医学问答与看图回答的研究试用。** 已完成训练和主要测试；本说明不代表服务当前在线。路径来自项目记录，执行前需在主机核对文件。无需等待 CheXbert 复核即可试问，但成人胸片描述的判断仍待复核。

**原版基座 / Base model（只读）**
```
/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
```

**训练补丁 / Trained adapters（以下是完整权重文件路径）**

| 版本 | 补丁权重文件 | 训练方式 |
|---|---|---|
| A42，默认试用 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_s2a/adapter_model.safetensors` | 八数据集问答微调，种子 42 |
| A43，重复实验 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_s2a_s43/adapter_model.safetensors` | 同样问答微调，种子 43 |
| B42 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_s2b/adapter_model.safetensors` | 知识预训练后问答微调，种子 42 |
| B43 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_s2b_s43/adapter_model.safetensors` | 同一个预训练补丁后问答微调，种子 43 |
| C42 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_s3c/adapter_model.safetensors` | B42 后再做答案与解析微调 |
| cpt1，仅预训练对照 | `/home/ubuntu/chunqian/MedLoRA/outputs/interns2_cpt1/adapter_model.safetensors` | 只做额外文字知识预训练，种子 42 |

合并时传入**权重所在目录**，目录还应包含 `adapter_config.json`。每次选择一个最终补丁；B、C 的补丁已包含前面阶段的训练结果，无需逐个叠加。43 重训的是问答微调阶段，未重复知识预训练。

**试用选择 / Trial choice**

先试 A42，并与原版回答同一批题。选择 A 是为了用较简单的训练方案检查实际表现，不表示它在所有任务上最好。C 只有单种子结果；不能因为多一个训练阶段就认定更好。

`Intern-S2-Preview-MedLoRA` 是早期两数据集部署版，**不是本页的 A/B/C**。临时合并模型已清理，最终补丁保留；重新合并不等于重新训练。
"""),
    ("执行前准备", "Before deployment", r"""
**运行位置：5090 主机，ubuntu 账号；命令使用 Bash。** 不在 user0 容器运行，不在本机加载 35B。需先获得 0–3 号卡的可用时段，并约定归还时间；4–7 号卡不可用。停止原服务仅交给 `gpu_window.sh`。

### 1. 同步本次新增的 demo 模式（本机 Git Bash）
服务器是代码副本；旧脚本不认识 `STEPS=demo`。只同步这个文件，原子替换以免覆盖正在读取的旧文件。以下操作只同步代码，不启动模型。
```bash
cd '/c/Users/Y4/fyp/Topic 6/MedLoRA'
tar -czf trial-update.tar.gz scripts/server/gpu_window.sh
scp trial-update.tar.gz 5090:/home/ubuntu/chunqian/
ssh 5090 'cd /home/ubuntu/chunqian/MedLoRA &&
tar -xOf ../trial-update.tar.gz scripts/server/gpu_window.sh \
  > scripts/server/gpu_window.sh.next &&
bash -n scripts/server/gpu_window.sh.next &&
mv scripts/server/gpu_window.sh.next scripts/server/gpu_window.sh &&
sha256sum scripts/server/gpu_window.sh'
sha256sum scripts/server/gpu_window.sh
```
两边 SHA-256 应相同。连接失败即停，先处理连接问题。

### 2. 核对补丁、GPU 和磁盘（主机 Bash）
```bash
cd /home/ubuntu/chunqian/MedLoRA
ls -lh outputs/interns2_s2a/adapter_model.safetensors
cat outputs/interns2_s2a/adapter_config.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv -i 0,1,2,3
df -h / /home/share /home/ubuntu/chunqian/merged
readlink -f /home/ubuntu/chunqian/merged
```
合并目录所在盘需**至少 80 GB 可用空间**。若需临时借用 `/home/share`，每次都要重新获得管理员同意；之前的许可不自动延续。

### 3. 仅在已获准借盘且 merged 为空时设置临时位置
```bash
R=/home/ubuntu/chunqian
test -d "$R/merged" && test ! -L "$R/merged"
ls -A "$R/merged"                 # 必须为空
test ! -e /home/share/chunqian_tmp
mkdir -m 700 /home/share/chunqian_tmp
rmdir "$R/merged" && ln -s /home/share/chunqian_tmp "$R/merged"
```
逐步核对，任一步失败就停；已有目录或链接时先核实，不能自行删除。空间充足且不需借盘时，跳过第 3 步。
"""),
    ("合并并限时启动", "Merge and start a timed trial", r"""
### 1. 选模型与截止时间
下面以 A42 为例。截止时间使用**主机北京时间（UTC+8）**，必须改成已约定的未来时间，并留出至少 15 分钟。一次只启动一个模型。
```bash
cd /home/ubuntu/chunqian/MedLoRA
DEADLINE_AT='YYYY-MM-DD HH:MM'       # 替换为已约定时间
REPO=/home/ubuntu/chunqian/MedLoRA
TRIAL="exec env DEADLINE_AT='$DEADLINE_AT' MODELS=a STEPS=demo"
tmux new-session -d -s medlora_demo \
  "$TRIAL bash $REPO/scripts/server/gpu_window.sh"
tail -20 /home/ubuntu/chunqian/logs/gpu_window.log
```
`MODELS` 可选：`a`＝A42，`a43`＝A43，`b`＝B42，`b43`＝B43，`c`＝C42，`cpt1`＝仅预训练。其他命令不变。

脚本先观察原服务两分钟；发现请求或 GPU 忙则退出。通过检查后停止原服务，合并指定补丁，启动试用服务。日志出现 **READY 人工试用** 后才开始提问。试用期间不会重新训练，也不会重跑整套评测。

**A42 合并后的模型目录**
```
/home/ubuntu/chunqian/merged/interns2_s2a
```
如 merged 指向已批准的临时盘，文件实际占用的是临时盘空间。脚本拒绝覆盖已有同名合并目录；不要为绕过检查自行删除旧文件。

### 2. 合并内部实际调用的命令（用于理解，不要重复运行）
```bash
bash train/interns2/run_sft35.sh merge a
PORT=23334 bash train/interns2/serve.sh start \
  /home/ubuntu/chunqian/merged/interns2_s2a
```
第一行在训练环境中加载原版基座和 A42 补丁，再保存完整模型；第二行通过 LMDeploy 使用 0–3 号卡启动接口。**人工试用应执行上面的时间窗命令**，由它统一负责启动、截止和恢复。

### 3. 结束与恢复
到点后自动结束试用，清理本次新建的临时合并模型，保留补丁，再启动原服务。合并或启动阶段每约 30 秒检查截止时间；恢复服务还需要启动时间。

需要提前结束时，在另一终端执行：
```bash
tmux send-keys -t medlora_demo C-c
tail -20 /home/ubuntu/chunqian/logs/gpu_window.log
```
等日志确认原服务已恢复，再交还使用时段。不要用强制杀进程或直接关闭整台机器的方式结束。恢复失败时查看日志并联系服务维护方。
"""),
    ("提问与对比", "Ask questions and compare answers", r"""
### 1. 在另一个主机 Bash 终端提问
```bash
cd /home/ubuntu/chunqian/MedLoRA
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/chunqian/envs/s2train
python scripts/compare_ask.py ask \
  '请用三句话解释什么是气胸，不使用难懂的术语。'
python scripts/compare_ask.py ask \
  'Explain pneumothorax in three sentences using plain language.'
```
若 conda 安装在其他位置，按主机实际安装位置加载 `conda.sh`。工具每次先查询当前模型 ID，并在输出中标记模型。A42 应显示 `interns2_s2a`，A43 应显示 `interns2_s2a_s43`；显示其他版本时先核对。

接口为主机上的 `http://localhost:23334/v1`，提供 `/models` 和 `/chat/completions`。保持服务器内部访问；已有界面接入时，也必须按 `/models` 的实际返回值确认版本。请求关闭思考模式，`temperature=0`，与已有评测一致。

### 2. 看图提问
```bash
IMAGE=/absolute/path/to/public_image.png
python scripts/compare_ask.py ask \
  'Describe this image in two sentences. State uncertainty when needed.' \
  --image "$IMAGE"
```
将图片路径换成**获准使用的公开教学图片**。可用带标准答案的 SLAKE / VQA-RAD 测试图片；使用时保留样本编号和标准答案。不要使用受控胸片、私人病人资料或未经许可的数据。无图提问不能用来评价看图能力。

### 3. 固定题目，保存回答
将第 5–6 页题目保存为 UTF-8 的 `outputs/compare/trial_questions.jsonl`，一行一道题。例如：
```json
{"id":"q01_zh","question":"请用三句话解释什么是气胸，不使用难懂的术语。"}
{"id":"q01_en","question":"Explain pneumothorax in three short sentences."}
```
```bash
mkdir -p outputs/compare
python scripts/compare_ask.py batch outputs/compare/trial_questions.jsonl \
  --out outputs/compare/trial_A42.jsonl
```
批量工具已有同 ID 回答会跳过。换题或改提示词时换一个输出文件，不把新旧回答混在一起。记录模型、完整问题、图片编号、回答和是否被截断。

**与原版比较：** 不能同时开两个 35B。先结束 A 的窗口并确认恢复；另约原版试用窗口，由服务维护方提供原版接口。使用同一题目文件和相同生成设置，保存到单独文件；工具可用 `--base <原版接口地址>` 指定接口。不能把早期两数据集版当作原版或 A。

**试用能发现具体问题；几道例题不能代替准确率。** 问答分数来自完整测试集。自由回答需有参考依据或专业复核，不能只因语气流畅就判为正确。
"""),
    ("建议题目：文字问答", "Suggested prompts: text questions", r"""
每组中英文分别提问，保持意思一致。这些是演示题，不是现有测试集的标准答案，也不保证训练时从未见过类似内容。观察点用于检查表达和推理，医学事实仍需可靠参考或专业复核。

### Q1 · 概念解释 / Plain-language explanation
**中文：** 请用三句话解释什么是气胸，不使用难懂的术语。

**English:** Explain pneumothorax in three sentences using plain language.

**观察：** 是否回答所问、用词清楚、遵守三句话要求；内容是否经得起参考资料核对。

### Q2 · 比较与格式 / Comparison and format
**中文：** 用三行表格比较 CT 和 MRI 的成像原理、是否使用电离辐射，以及一个常见用途。不要说其中一种在所有情况下都更好。

**English:** Use a three-row table to compare CT and MRI by imaging principle, use of ionizing radiation, and one common use. Do not claim that either is better in every situation.

**观察：** 三个方面是否都覆盖；是否按表格回答；是否把复杂差异说成绝对结论。

### Q3 · 信息不足 / Recognizing insufficient information
**中文：** 虚构练习：一个人咳嗽、发热两天。仅凭这些信息能确定是肺炎吗？请说明还缺哪些信息，不要开药。

**English:** Fictional exercise: a person has had a cough and fever for two days. Is this information alone enough to confirm pneumonia? Explain what information is missing. Do not prescribe medication.

**观察：** 是否承认信息不足、区分可能性与确定诊断、遵守不提供处方的要求。

### Q4 · 阅读证据 / Reading evidence
**中文：** 虚构练习：一项研究报告“服用 X 的人平均恢复更快”，但没有说明是否随机分组，也没有给出两组病情。仅凭这段话能证明 X 导致恢复更快吗？只回答“能”或“不能”，再用一句话说明原因。

**English:** Fictional exercise: a study says that people taking X recovered faster, but gives no information about randomization or baseline illness severity. Does this statement alone prove that X caused faster recovery? Answer “yes” or “no”, then explain in one sentence.

**观察：** 是否依照给定信息判断；是否避免编造样本数、疗效或不存在的研究细节；是否符合输出格式。
"""),
    ("建议题目：图片与多轮", "Suggested prompts: images and follow-up", r"""
### Q5 · 图片短答 / Short visual answer
对同一张有标准答案的公开医学教学图片分别提问：

**中文：** 这张图片是什么影像检查类型？只回答检查类型；无法判断时回答“不确定”。

**English:** What imaging modality is shown? Answer with the modality only. If it cannot be determined, answer “uncertain”.

**观察：** 与该图片的标准答案核对，不能事先假定图片必然是 X 光或 CT。不要仅凭模型自信程度打分。

### Q6 · 短答与描述一致性 / Consistency between answers and descriptions
选择获准使用、具有异常标注的公开胸片。以下两问分别附上**同一张图**，以独立请求提问；中英文各做一遍。

**中文①：** 这张胸片是否可见异常？只回答“是”“否”或“不确定”。

**English ①:** Is any abnormality visible in this chest X-ray? Answer only “yes”, “no”, or “uncertain”.

**中文②：** 请用两到三句话描述这张胸片，包括可见的异常；看不清或不能确定的地方请明确说明。

**English ②:** Describe this chest X-ray in two or three sentences, including visible abnormalities. Explicitly state anything that is unclear or uncertain.

**观察：** 是否一边回答“有异常”、一边又描述“完全正常”；是否遗漏标注中的异常；是否添加图中没有的异常。两个回答一致仍可能一起答错，必须结合标注复核。此处人工试问不能替代待完成的 CheXbert 复核。

### Q7 · 多轮衔接 / Follow-up conversation
本组在支持对话历史的界面中测试；单独重复运行 `compare_ask.py ask` 不会保留上一轮。

**中文①：** 请用一句话解释“影像上的异常不一定等于已经确诊”。

**中文②：** 把刚才的解释改成更适合没有医学背景的人理解的说法，仍然只用一句话。

**English ①:** Explain in one sentence why an abnormal imaging finding does not by itself establish a diagnosis.

**English ②:** Rephrase your previous explanation for someone without a medical background, still using only one sentence.

**观察：** 是否记住上一轮内容、保留原意、按要求变得更易懂。界面需把两轮历史一并发送。

**记录方式：** 每题记录“符合 / 部分符合 / 不符合”，另标事实错误、漏答、矛盾、无依据细节或格式问题。少量试用记录只作定性比较，不能据此宣称临床可用。
"""),
    ("结束检查与文件说明", "Shutdown checks and reference files", r"""
### 1. 确认已恢复原服务
```bash
tail -30 /home/ubuntu/chunqian/logs/gpu_window.log
ss -ltn | grep -E ':2333[34] '
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv -i 0,1,2,3
```
正常结束后，日志应确认原服务已恢复，23333 应监听，试用服务 23334 应停止。原服务启动后重新占用 GPU 显存是预期行为。日志未确认恢复时不要直接离开。

### 2. 若本次借用了临时盘，核对后归还
demo 模式仅清理本次生成的完整模型，永久补丁不删。下列操作只用于本说明创建的临时链接，**先检查链接目标和目录为空**；不匹配就停止，不删除其他目录。
```bash
R=/home/ubuntu/chunqian
readlink "$R/merged"              # 应为 /home/share/chunqian_tmp
ls -A /home/share/chunqian_tmp      # 必须为空
if [ -L "$R/merged" ] &&
   [ "$(readlink "$R/merged")" = /home/share/chunqian_tmp ]; then
    if rmdir /home/share/chunqian_tmp; then
        unlink "$R/merged"
        mkdir "$R/merged"
    fi
fi
```
`rmdir` 只删除空目录；非空时保留现场并核实。测试回答文件保留在 `outputs/compare/`，不要把真实病人资料、受控图片或受控逐条记录带出服务器。

### 3. 本说明涉及的代码和记录

| 文件 | 用途 |
|---|---|
| `scripts/server/gpu_window.sh` | 本次新增 `STEPS=demo`，统一限时与恢复 |
| `train/interns2/run_sft35.sh` | A/B/C 与 43 版本的映射、合并检查 |
| `train/interns2/merge_lora.py` | 基座与一个最终补丁合并 |
| `train/interns2/serve.sh` | 启动与停止本项目的模型服务 |
| `scripts/compare_ask.py` | 单题、图片题、批量提问与保存回答 |
| `results/README.md`、`docs/report.md` | 已完成的实验结果与解读 |

**状态边界：** 本说明根据本机代码与实验记录编写。未在服务器核对当前文件是否仍在、服务是否在线，也未实际执行新 demo 模式。部署前必须完成补丁、空间、脚本版本和使用时段检查。

**使用范围：** 用于研究演示和回答质量检查。现有分数不能证明任意医学问题都能正确回答；试用结果也不能代替专业判断。
"""),
]

CSS = """
@page { size:A4; margin:0; }
* { box-sizing:border-box; }
html { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body { margin:0; color:#182c3b; font-family:'Microsoft YaHei',Arial,sans-serif; }
.page { width:210mm; height:297mm; padding:16mm 17mm 18mm; position:relative; break-after:page; }
.page:last-child { break-after:auto; }
.kicker { color:#187b85; font-size:9pt; letter-spacing:1pt; border-top:3pt solid #187b85; padding-top:6pt; }
h1 { font-size:22pt; line-height:1.2; margin:9pt 0 3pt; }
.subtitle { color:#597181; font-size:11pt; margin-bottom:12pt; }
h3 { font-size:11pt; margin:11pt 0 4pt; }
p,li { font-size:9pt; line-height:1.55; margin:5pt 0; }
table { width:100%; border-collapse:collapse; table-layout:fixed; margin:8pt 0; }
th,td { border:0.6pt solid #c9d7df; padding:5pt; text-align:left; vertical-align:top; font-size:8pt; line-height:1.45; overflow-wrap:anywhere; }
th { background:#e9f2f4; }
.page:first-child th:nth-child(1) { width:19%; }
.page:first-child th:nth-child(2) { width:55%; }
.page:first-child th:nth-child(3) { width:26%; }
code { font-family:Consolas,'Microsoft YaHei',monospace; font-size:8pt; overflow-wrap:anywhere; }
td code { font-size:7.2pt; }
pre { background:#f0f5f7; border-left:2.5pt solid #91bac2; padding:7pt 8pt; margin:7pt 0; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-all; }
pre code { font-size:7.5pt; line-height:1.5; }
.footer { position:absolute; bottom:10mm; left:17mm; right:17mm; border-top:0.6pt solid #c9d7df; padding-top:5pt; color:#597181; font-size:8pt; display:flex; justify-content:space-between; }
"""


def main():
    render = mistune.create_markdown(plugins=["table"], escape=False)
    pages = []
    for number, (title, subtitle, markdown) in enumerate(PAGES, 1):
        pages.append(
            f'<section class="page"><div class="kicker">MEDLORA · MODEL TRIAL GUIDE</div>'
            f'<h1>{title}</h1><div class="subtitle">{subtitle}</div>'
            + render(markdown)
            + f'<div class="footer"><span>Intern-S2-Preview · 35B · 2026-10-29</span>'
            f'<span>{number} / {len(PAGES)}</span></div></section>'
        )
    HTML.parent.mkdir(exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                    f'<title>MedLoRA model trial guide</title><style>{CSS}</style></head><body>'
                    + "".join(pages) + '</body></html>', encoding="utf-8")
    before = PDF.stat().st_mtime_ns if PDF.exists() else None
    result = subprocess.run(
        [str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--user-data-dir=" + str(REPO / "outputs/edge_trial_profile"),
         "--print-to-pdf=" + str(PDF), HTML.as_uri()],
        capture_output=True, text=True, timeout=180,
    )
    if not PDF.exists() or PDF.stat().st_mtime_ns == before:
        raise SystemExit("PDF 未生成: " + result.stderr[-500:])
    print(f"written {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
