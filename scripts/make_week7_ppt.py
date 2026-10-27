"""第 7 周周进展汇报 PPT (课题组统一模板 Weekly Report_Templet): docs/weekly/week7/Topic6_WeeklyReport7.pptx
做法同 make_week6_ppt.py: 只往模板原有的文本框 / 表格里填字, 每格「中文一行 + 英文一行」; 数字从 results/ 的快照读。
种子 43 的结果放在 results/interns2_seed43_eval_*.json (最新一份); 还没有就显示「训练中 / 待评测」, 有了重跑本脚本即可。
用法: python scripts/make_week7_ppt.py <模板.pptx> [输出.pptx]   (输出默认 docs/weekly/week7/Topic6_WeeklyReport7.pptx; 讲稿写在输出旁边)
"""
import copy
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
TEMPLATE = Path(sys.argv[1])
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else REPO / "docs/weekly/week7/Topic6_WeeklyReport7.pptx"

EV = json.loads((R / "interns2_3stage_eval_2026-10-19.json").read_text(encoding="utf-8"))["models"]
TR = json.loads((R / "interns2_3stage_train_2026-10-18.json").read_text(encoding="utf-8"))
CB = json.loads((R / "interns2_chexbert_2026-10-18.json").read_text(encoding="utf-8"))
SIX, MA, MB = (EV[k] for k in ("S2-6datasets", "A (②)", "B (①→②)"))
S43_FILES = sorted(R.glob("interns2_seed43_eval_*.json"))
S43 = json.loads(S43_FILES[-1].read_text(encoding="utf-8"))["models"] if S43_FILES else {}
A43, B43 = S43.get("interns2_s2a_s43", {}), S43.get("interns2_s2b_s43", {})
EVALUATED = bool(A43.get("slake_closed") is not None and B43.get("slake_closed") is not None)
# S2-CheXpertPlus (只练胸片报告): 问答与是非题来自 10-13 快照, CheXbert 漏报来自带回本机的容器汇总 (只有比例)
_cxp = json.loads((R / "interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]["S2-CheXpertPlus"]
CXP = {"slake_closed": _cxp["slake_closed"], "false_alarm": 100 - _cxp["chexpert"]["closed"]["specificity"],
       "chexbert_omission": json.loads((REPO / "outputs/eval/chexbert_interns2_cxp.json").read_text(encoding="utf-8"))
       ["abnormal_no_finding_reported_pct"]}
CI_FILES = sorted(R.glob("interns2_ci_seed43_BvsA_*.json"))
CI43 = json.loads(CI_FILES[-1].read_text(encoding="utf-8"))["tests"] if CI_FILES else {}


def r1(x):
    return str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def loss_of(tag):
    """训练损失后 10% 平均: 种子 42 从训练快照读, 种子 43 从种子 43 快照读; 没有就返回 None。"""
    m = next((v for k, v in TR["models"].items() if k.split(" ")[0] == tag), None)  # 键是「标签 (说明)」
    if m is not None:
        return m.get("train_loss_last10pct_mean")
    return S43.get(tag, {}).get("train_loss_last10pct_mean")


def status(m):
    if m.get("slake_closed") is not None:
        return None
    return ("训练完成，待评测", "Trained, eval pending") if m.get("train_loss_last10pct_mean") else ("训练中", "Training")


# ---------------------------------------------------------------- 填字工具 (与 make_week6_ppt.py 相同)
def fill(tf, lines):
    """把文本框 / 单元格的段落换成 lines (每项一段); 段落不够就复制最后一个非空段落, 多了就删掉。"""
    paras = list(tf.paragraphs)
    proto = next((p for p in reversed(paras) if p.runs), paras[0])
    for p in paras[len(lines):]:
        p._p.getparent().remove(p._p)
    paras = paras[:len(lines)]
    while len(paras) < len(lines):
        el = copy.deepcopy(proto._p)
        paras[-1]._p.addnext(el)
        paras = list(tf.paragraphs)
    for p, text in zip(tf.paragraphs, lines):
        runs = p.runs
        if not runs:
            if text == "":
                continue
            p._p.append(copy.deepcopy(proto.runs[0]._r))
            runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def table(slide):
    return next(s for s in slide.shapes if s.has_table).table


def set_rows(tbl, n_body, row_h=None):
    rows = tbl._tbl.tr_lst
    while len(rows) - 1 > n_body:
        rows[-1].getparent().remove(rows[-1])
        rows = tbl._tbl.tr_lst
    while len(rows) - 1 < n_body:
        rows[-1].addnext(copy.deepcopy(rows[-1]))
        rows = tbl._tbl.tr_lst
    if row_h:
        for r in tbl.rows:
            r.height = Emu(int(Inches(row_h)))


def fill_table(tbl, body):
    for i, row in enumerate(body, start=1):
        for j, lines in enumerate(row):
            fill(tbl.cell(i, j).text_frame, list(lines))


def boxes(slide, items, takeaway):
    for i, body_lines in enumerate(items):
        fill(shape(slide, f"body-{i}").text_frame, body_lines)
    fill(shape(slide, "takeaway").text_frame, takeaway)


CB_BY_TAG = {v["tag"]: v["abnormal_no_finding_reported_pct"] for v in CB["models"].values()}
CB_R0 = [CB_BY_TAG[k] for k in ("interns2_mix_0", "interns2_abl_r0_s43")]  # 回放 0 两个种子

prs = Presentation(str(TEMPLATE))
S = prs.slides
NOTES = []

# ---------------------------------------------------------------- 1 封面
fill(shape(S[0], "details").text_frame, [
    "医学多模态模型的参数高效增量预训练与指令微调",
    "Parameter-Efficient Incremental Pretraining and Instructional Fine-Tuning of Medical Multimodal Models",
    "LOO CHUN QIAN    导师 / Supervisor：徐蕾",
    "第 7 周 / Week 7    2026-10-29",
    "",
    "本周核心进展：种子 43 的 A、B " + ("已训完并评完" if EVALUATED else "已训完") + "；英文报告补到 v1.6；毕设表 1–3 草稿",
    "Main update: seed-43 A and B " + ("trained and evaluated" if EVALUATED else "trained") + "; report v1.6; thesis Tables 1–3 drafted",
])
NOTES.append("这周三件事：第一，上周暂停的种子 43 重训，世龙 10 月 24 日让出显卡，A 和 B 都" + ("训完并评完了。" if EVALUATED else "训完了，评测在等硬盘空间。")
             + "第二，英文报告补到 1.6 版，共 37 页，种子 43 对比和配对区间已写入。第三，表 1 到 3 的草稿已经写好，学校的原始文件还包含空白的表 4 指导记录。")

# ---------------------------------------------------------------- 2 本周进展 (上周承诺)
t = table(S[1])
fill_table(t, [
    [("种子 43 重训 A、B", "Seed-43 retrain of A and B"),
     (("完成：训完、评完", "Done: trained and evaluated") if EVALUATED else ("训练完成，评测待做", "Trained; eval pending")),
     (("结果快照已推 GitHub", "Snapshot pushed to GitHub") if EVALUATED else ("合并模型还缺约 30 GB 硬盘", "Merge needs ~30 GB more disk"))],
    [("把结果写进英文报告", "Write results into the report"),
     ("完成：v1.6，37 页", "Done: v1.6, 37 pages"),
     ("种子 43 对比与配对区间已补", "Seed-43 comparison and paired CIs added")],
    [("CheXbert 复核 A、B、C", "CheXbert re-check of A, B, C"),
     ("安排到下周", "Planned for next week"),
     ("容器 CPU 共用，执行前先征得同意", "Shared CPU; get agreement before running")],
])
fill(shape(S[1], "takeaway").text_frame, [
    "已解决：合并空间已归还；报告已补种子 43    下周：队友同意后做 CheXbert 复核",
    "Done: merge space returned; seed 43 added to report    Next week: CheXbert once teammates agree",
] if EVALUATED else [
    "主要阻碍：系统盘只剩约 49 GB，合并一个模型要 75 GB    所需支持：评测时临时用 /home/share",
    "Main blocker: ~49 GB free, a merge needs 75 GB    Support needed: temporary use of /home/share",
])
NOTES.append("上周承诺的三件事：种子 43 重训" + ("已经训完、评完，系统盘不够的问题，苏学长同意临时借 /home/share 放合并模型，评完已经删掉；" if EVALUATED else "已经训完，但评测要先把模型合并成一个 73 GB 的文件，系统盘现在只剩约 49 GB，放不下；")
             + "英文报告已经写到 1.6 版，共 37 页；CheXbert 复核安排到下周，容器的 CPU 和队友共用，执行前要先征得同意。"
             + ("" if EVALUATED else "需要的支持：评测时临时借用 /home/share 的空间，评完就删。"))

# ---------------------------------------------------------------- 3 核心挑战与研究问题
boxes(S[2], [
    ["输入医学图像 + 中英文问题，输出短答或描述；35B，LoRA 只训 0.054%",
     "Image + EN/ZH question → answer or description; 35B, LoRA 0.054%"],
    ["同一配方换个种子，胸片描述漏报能差 34 点；单次训练的结论不可靠",
     "Same recipe, another seed: X-ray omission can differ by 34 points"],
    ["RQ2：A 的胸片描述比 S2-6datasets 差，是中文对话造成的还是种子造成的？",
     "RQ2: is A's worse X-ray description caused by the dialogues or by the seed?",
     "本周验证：A、B 用种子 43 重训 / This week's test: retrain A and B with seed 43"],
], [
    "判断标准：种子 43 的 A 若回到 S2-6datasets 水平（约 11–14%）→ 种子；若仍约 20% → 中文对话",
    "Criterion: if seed-43 A returns to ~11–14% → seed; if it stays near 20% → the dialogue data",
])
NOTES.append("上周发现：A 只比 S2-6datasets 多了中文对话，胸片描述漏报却从 13.8% 变成 19.8%。原因有两种可能：中文对话的影响，或者只是随机种子。"
             "同一配方换种子能差 34 点，所以必须再训一次。判断标准事先定好：种子 43 的 A 如果回到 11 到 14%，就是种子；如果还在 20% 左右，就是中文对话。")

# ---------------------------------------------------------------- 4 文献
t = table(S[3])
fill_table(t, [
    [("Medical Adaptation…Progress?", "Jeong et al., EMNLP 2024"),
     ("医学版多数不稳定超过原版", "Medical models rarely beat base"),
     ("只测零样本 / 少样本，未测微调后", "Zero/few-shot only, not after SFT")],
    [("Fine-Tuning Pretrained LMs", "Dodge et al., arXiv 2002.06305"),
     ("初始化、数据顺序影响相当", "Init ≈ data order in variance"),
     ("只做了文本分类，没测生成", "Text classification only")],
    [("CheXbert", "Smit et al., EMNLP 2020"),
     ("从报告自动标 14 类发现", "Labels 14 findings in reports"),
     ("为医生报告设计，非模型描述", "Built for radiologist reports")],
])
fill(shape(S[3], "takeaway").text_frame, [
    "研究空白：参数高效微调后，自由描述的可靠性随种子怎么变    本周阅读：3 篇，数字均与原文核对",
    "Gap: how free-text reliability varies across seeds after PEFT    Reading: 3 papers, numbers checked",
])
NOTES.append("三篇文献：第一篇 EMNLP 2024 比了七个医学大模型和两个医学多模态模型，3 样本设置下医学版大模型只在 12.1% 的情况显著超过原版，"
             "和我们预训练没用的结果一致。第二篇说只换随机种子，初始化和数据顺序对结果波动的贡献差不多，这就是我们重训种子 43 的理由。"
             "第三篇是 CheXbert，我们用它代替关键词规则判描述；它是为医生报告设计的，所以我们先在 300 份医生报告上校验，一致率 96.2%。")

# ---------------------------------------------------------------- 5 拟议方法
boxes(S[4], [
    ["同一配方 × 两个种子 → 配对区间 + CheXbert → 只报两次都成立的结论",
     "Recipe × 2 seeds → paired CIs + CheXbert → keep findings both seeds show"],
    ["多数医学模型工作只训一次；我们每个结论都要求两个种子 + 区间",
     "Most medical-VLM work trains once; we require two seeds and an interval"],
    ["C1：35B 上预训练的受控消融；C2：可靠性评测（漏报、矛盾、去图）",
     "C1: CPT ablation at 35B; C2: reliability (omission, contradiction, no image)",
     "本周：评测流程不变，只换种子 / This week: same evaluation, new seed"],
], [
    "胸片描述：本周先报关键词结果；下周 CheXbert 复核后以其为准",
    "X-ray descriptions: keyword results now; CheXbert re-check next week",
])
NOTES.append("方法上这周没有新东西，重点是把结论做扎实：同一配方训两个种子，主表看配对置信区间。胸片描述本周仍是关键词判断，下周经队友同意再用 CheXbert 复核，复核后以 CheXbert 为准。")

# ---------------------------------------------------------------- 6 实验计划
t = table(S[5])
fill_table(t, [
    [("问题与对比", "Question / comparison"),
     ("A、B 各用种子 42 和 43；看 B − A 和胸片描述是否两次一致",
      "A and B with seeds 42 and 43; is B − A and the X-ray result stable?")],
    [("数据与划分", "Data / split"),
     ("与种子 42 完全相同：六个看图数据集 + 华佗、DISC 中文对话 27,953 条",
      "Same data as seed 42: six VQA sets + Huatuo, DISC dialogues, 27,953")],
    [("指标与控制变量", "Metrics / controls"),
     ("看图 / 考题准确率（↑）；胸片漏报本周关键词、下周 CheXbert（↓）",
      "VQA / exam accuracy (↑); omission: keywords now, CheXbert next week (↓)")],
    [("消融与成功标准", "Ablation / criterion"),
     ("结论两种子都成立才算；B 仍从种子 42 的预训练补丁接着训",
      "A finding counts only if both seeds agree; B starts from the seed-42 CPT adapter")],
])
fill(shape(S[5], "takeaway").text_frame, [
    "本周实验：A43 从第 2,500 步续训，B43 接着训；10-25 08:30 评完，自动恢复世龙服务",
    "This week's run: A43 resumed at step 2,500, then B43; evaluated by Oct 25, service restored",
] if EVALUATED else [
    "本周实验：A43 从第 2,500 步续训，B43 接着训；截止 10-25 14:00 自动恢复世龙服务",
    "This week's run: A43 resumed at step 2,500, then B43; Shilong's service restored automatically",
])
NOTES.append("实验设计：数据和设置与种子 42 完全一样，只换种子，种子决定 LoRA 初始值和数据顺序。B 仍然从种子 42 的预训练补丁接着训，"
             "所以这一轮比较的是微调这一步的波动。A43 从上次暂停的第 2,500 步接着训，接力脚本到点会自动停下并开回世龙的服务。")

# ---------------------------------------------------------------- 7 实验结果
t = table(S[6])
set_rows(t, 5 if EVALUATED else 4, row_h=0.6 if EVALUATED else 0.72)


def row(name, m, tag):
    st = status(m)
    loss = loss_of(tag)
    return [name,
            (f"{r1(m['slake_closed'])} / {r1(m['slake_open_em'])}", "SLAKE closed / open") if not st else st,
            (f"{r1(m['cxr_desc_miss_pct'])}%", "X-ray omission") if not st else ("—", "pending"),
            (tag, f"loss {loss:.3f}" if loss else "training")]


fill_table(t, ([
    [("S2-6datasets（参照）", "S2-6datasets (reference)"),
     (f"{r1(SIX['slake_closed'])} / {r1(SIX['slake_open_em'])}", "SLAKE closed / open"),
     (f"{r1(SIX['cxr_desc_miss_pct'])}%", "X-ray omission"), ("interns2_6ds", "no dialogues")],
] if EVALUATED else []) + [
    row(("A 种子 42（上周）", "A, seed 42 (last week)"), MA, "interns2_s2a"),
    row(("A 种子 43（本周）", "A, seed 43 (this week)"), A43, "interns2_s2a_s43"),
    row(("B 种子 42（上周）", "B, seed 42 (last week)"), MB, "interns2_s2b"),
    row(("B 种子 43（本周）", "B, seed 43 (this week)"), B43, "interns2_s2b_s43"),
])
if EVALUATED:
    a43 = r1(A43["cxr_desc_miss_pct"])
    take = [f"A43 漏报 {a43}%，种子差约 4 点，对话影响未定；关键词判断，下周 CheXbert 复核",
            f"A43 omission {a43}%, seed shifts ~4 pts; dialogue effect unclear. Keyword judge; CheXbert next week"]
elif A43.get("train_loss_last10pct_mean") and B43.get("train_loss_last10pct_mean"):
    take = ["观察：两个种子的训练损失几乎相同    解释：评测还没做，结论待定    局限：缺硬盘合并模型",
            "Observation: training losses match across seeds    Interpretation: pending evaluation    Limitation: disk"]
else:
    take = ["观察：A43、B43 训练进行中    解释：评测后再下结论    局限：缺硬盘合并模型",
            "Observation: A43 and B43 still training    Interpretation: after evaluation    Limitation: disk"]
fill(shape(S[6], "takeaway").text_frame, take)
def ci(test, key="interns2_s2b_s43"):
    v = CI43[test][key]
    return f"{v['diff_vs_ref']:+.1f}（区间 {v['diff_ci95'][0]:+.1f} 到 {v['diff_ci95'][1]:+.1f}）"


NOTES.append("结果：" + (
    f"先回答上周的问题。按事先定的标准，种子 43 的 A 胸片描述漏报是 {r1(A43['cxr_desc_miss_pct'])}%，"
    f"种子 42 是 {r1(MA['cxr_desc_miss_pct'])}%，没有中文对话的 S2-6datasets 是 {r1(SIX['cxr_desc_miss_pct'])}%。"
    "两个条件都不完全满足，落在中间：只换种子就能差约 4 点，所以上周看到的 6 点差距至少有一部分是种子造成的，"
    "现在不能说中文对话让胸片描述变差。这些数字是关键词判断的，下周经队友同意后用 CheXbert 复核两个种子。"
    "其次，B 减 A 在两个种子下方向基本一致：SLAKE 是非题两次都低 0.7 点，但区间跨 0；PathVQA 是非题两次都低不到 1 点，"
    f"种子 43 是 {ci('PathVQA closed')}；胸片描述漏报 B 两次都高约 2.7 点。所以「文字预训练没有可测提升」在两个种子下都成立。"
    f"其他看图、考题分数，同一个模型换种子的差大多在 2 点以内，最大的是 B 的儿童肺炎题，差 {r1(B43['pneumonia_closed'] - MB['pneumonia_closed'])} 点；"
    f"训练损失也几乎相同（A 是 {loss_of('interns2_s2a'):.3f} 和 {loss_of('interns2_s2a_s43'):.3f}）。"
    if EVALUATED else
             f"A 和 B 用种子 43 都训完了，训练损失和种子 42 几乎一样。评测还没做，因为要先合并模型、硬盘不够。"
             f"参照：同一配方的回放 0 两个种子，CheXbert 判的描述漏报是 {r1(CB_R0[0])}% 和 {r1(CB_R0[1])}%，"
             "所以评测出来之前不下结论。"))

# ---------------------------------------------------------------- 8 预期成果与阶段结论
boxes(S[7], [
    ["英文报告 v1.6（37 页，种子对比在第 28 页）；毕设表 1–3 草稿",
     "Report v1.6 (37 pp., seed comparison on p. 28); Tables 1–3 drafted"],
    ["已确认（两个种子）：分数来自微调；文字预训练没有可测提升",
     "Two SFT seeds: gains from fine-tuning; no measurable gain from CPT"] if EVALUATED else
    ["已确认：分数来自微调；文字预训练、推理训练没有可测提升（种子 42）",
     "Established: gains from fine-tuning; CPT, reasoning add nothing (seed 42)"],
    ["待定：关键词漏报换种子差约 4 点，对话影响未定；下周 CheXbert 复核",
     "Seed shifts omission ~4 pts; dialogue effect unclear; CheXbert next week"] if EVALUATED else
    ["待定：胸片描述变差是中文对话还是种子 → 等种子 43 评测",
     "Open: dialogue data or seed behind the X-ray drop → seed-43 evaluation"],
], [
    "最终交付：代码（GitHub MedLoRA）、LoRA 补丁、数据卡 / 模型卡、实验评估、论文",
    "Final deliverables: code, LoRA adapters, data / model cards, experimental evaluation, thesis",
])
NOTES.append("阶段结论：分数来自微调、文字预训练没有可测提升，英文报告已更新到 1.6 版，共 37 页，种子对比表在第 28 页。"
             + ("种子 43 也验证了这一点：预训练没有可测提升在两个 SFT 种子下都成立；B43 沿用种子 42 的预训练补丁。还没定的是胸片描述：关键词漏报换种子就能差约 4 点，中文对话有没有影响，要等下周 CheXbert 复核后再说。"
                if EVALUATED else "还没定的是胸片描述变差的原因，要等种子 43 评测。")
             + "毕设表 1 到 3 草稿写好了，要和老师约时间一起定稿签字。")

# ---------------------------------------------------------------- 9 时间安排
t = table(S[8])
fill_table(t, [
    [("第1–4周", "W1–4"), ("数据与基线：3B、35B 基座", "Data and baselines: 3B, 35B"),
     ("results/ 快照、周报 1–3", "Result snapshots, reports 1–3"), ("完成", "Done")],
    [("第5–8周", "W5–8"), ("35B 微调、三步训练", "35B fine-tuning, 3 stages"),
     ("评测结果、置信区间、报告 v1.6", "Results, CIs, report v1.6"), ("完成", "Done")],
    [("第9–12周", "W9–12"), ("种子 43、CheXbert、表 1–3", "Seed 43, CheXbert, Tables 1–3"),
     ("两种子结论、开题报告", "Two-seed findings, proposal"), ("进行中", "In progress")],
    [("第13–16周", "W13–16"), ("补充实验与论文", "Extra experiments, thesis"),
     ("误差分析、论文初稿", "Error analysis, draft"), ("未开始", "Not started")],
])
fill(shape(S[8], "takeaway").text_frame, [
    "下周计划：CheXbert 复核（队友同意后）；老师同意则准备 A + 胸片报告混训；表 1–3 定稿",
    "Next week: CheXbert once agreed; if approved, prepare A + CheXpert Plus mix; finalise Tables 1–3",
] if EVALUATED else [
    "下周承诺：硬盘解决后评测 A43、B43（约 4 小时）；表 1–3 与老师定稿（12-11 截止）",
    "Next week: evaluate A43 and B43 once disk allows (~4 h); finalise Tables 1–3 (due Dec 11)",
])
NOTES.append("时间安排：前八周的内容都完成了。" + (
    "种子 43 也训完、评完了。CheXbert 安排到下周：先问队友，同意后只用容器 CPU，优先复核两个种子的 A 和 B，再做 C 和只做第①步的补丁。"
    "如果老师同意把胸片报告混进问答微调，下周先准备数据和训练方案，等 CheXbert 结果出来、约好显卡再训。"
    "表 1 到 3 和老师定稿，学校 12 月 11 日截止；表 4 按实际指导内容持续记录，由老师确认。"
    if EVALUATED else
    "接下来是种子 43 评测和 CheXbert 复核，以及开题报告。下周承诺：硬盘一解决就评测 A43、B43，大约 4 小时显卡；表 1 到 3 和老师定稿，学校 12 月 11 日截止。"))

# ---------------------------------------------------------------- 10 讨论与决策
boxes(S[9], [
    ["① 要不要加种子 44？A：两个种子够，写进局限；B：再训一个 A（约 12 小时）",
     "① Seed 44? A: two seeds, state as a limit; B: one more A (~12 GPU h)",
     "② 胸片报告混进问答微调一起训？A：CheXbert 后再定；B：下周起准备",
     "② Mix CheXpert Plus reports into SFT? A: after CheXbert; B: prepare now"],
    ["下周 CheXbert 复核先问队友；表 2、表 3 约老师定稿；表 4 持续记录",
     "Next week: ask before CheXbert; finalise Tables 2–3; keep Table 4 updated"],
] if EVALUATED else [
    ["评测时合并模型能否临时放 /home/share？A：可以（评完即删）；B：等扩容",
     "Merged model on /home/share? A: yes, deleted after; B: wait for expansion"],
    ["约 4 小时显卡评测；表 2 任务书、表 3 开题报告约时间与老师一起填",
     "~4 GPU hours for evaluation; a meeting to fill Tables 2 and 3 together"],
    ["会议决策：（会后填写）", "Decision: (to fill after the meeting)", "执行事项：（会后填写） / Action: (after the meeting)"],
], [
    "每周材料：汇报 PPT、实验日志（results/README.md）、英文报告（docs/report.pdf）、代码（GitHub）",
    "Weekly package: slides, experiment log, report (docs/report.pdf), code (GitHub AugustLoo/MedLoRA)",
])
NOTES.append(("想请老师决定两件事。第一，胸片描述这个问题，两个种子的结果落在中间，要不要再加一个种子 44 的 A 来定。"
              "A 方案是两个种子就够，在论文里写成局限；B 方案是再训一个，大约 12 小时显卡，要再请世龙让卡。"
              f"第二，要不要把 CheXpert Plus 的胸片报告混进问答微调一起训。之前只用胸片报告训的 S2-CheXpertPlus，"
              f"胸片上是所有模型里最平衡的：CheXbert 判描述漏报 {r1(CXP['chexbert_omission'])}%，是非题误报 {r1(CXP['false_alarm'])}%，"
              f"但 SLAKE 是非题回到 {r1(CXP['slake_closed'])}，接近基座，A 是 {r1(MA['slake_closed'])}。"
              "混在一起训，是想两种能力都保住；但能不能做到要实验才知道，不一定更好。"
              "做法是：先做完 CheXbert 复核，看 A 和 B 的描述问题到底多大；再用 A 的数据加一部分胸片报告训练，两个种子，"
              "同时看胸片漏报、误报、自相矛盾和 SLAKE 等问答成绩。报告只用训练划分，和测试的 3,189 张片子不是同一批病人；"
              "数据只在服务器上用。每个种子大约 12 小时以上显卡。"
              if EVALUATED else "想请老师和苏学长决定：评测时能不能临时用 /home/share 放合并模型，评完马上删。")
             + "另外想约老师时间一起填表 2 任务书和表 3 开题报告。")

for slide, note in zip(S, NOTES):
    slide.notes_slide.notes_text_frame.text = note

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
TITLES = ["封面", "本周进展", "核心挑战与研究问题", "现有文献与研究空白", "拟议方法与独特贡献",
          "实验计划", "实验结果", "预期成果与阶段结论", "时间安排", "讨论与决策"]
script = ["# 第 7 周汇报讲稿（2026-10-29，约 5 分钟）", "",
          "与 PPT 备注相同；每页讲一段，大白话，只讲结果。", ""]
for i, (title, note) in enumerate(zip(TITLES, NOTES), start=1):
    script += [f"## {i}. {title}", "", note, ""]
OUT.with_name("Topic6_WeeklyReport7_Script.md").write_text("\n".join(script), encoding="utf-8")
print("written", OUT, "| seed-43 evaluated:", EVALUATED)
