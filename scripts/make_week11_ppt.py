"""第 11 周周进展汇报 PPT (课题组统一模板): docs/weekly/week11/Topic6_WeeklyReport11.pptx
内容 = 执行方案第 1–2 周 (补种子 + 三个对照) 的两道关结果, 按第 9 周汇报定的规则如实讲 (用户 11-29 选定)。
做法同 make_week7_ppt.py: 只往模板原有的文本框 / 表格里填字, 每格「中文一行 + 英文一行」; 数字全部从 results/ 的快照读。
成人胸片漏报是关键词规则判的 (CheXbert 复核未做); 第 9 周 PPT 里的 25.0 / 56.5 是 CheXbert 口径, 两者不能直接比。
用法: python scripts/make_week11_ppt.py docs/weekly/Weekly_Report_Template.pptx [输出.pptx]
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
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else REPO / "docs/weekly/week11/Topic6_WeeklyReport11.pptx"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))["models"]


OCT, C1127, C1128 = (load(f) for f in ("interns2_chexpert_2026-10-07.json", "interns2_chexpert_controls_2026-11-27.json",
                                       "interns2_chexpert_r300controls_2026-11-28.json"))
M = {"base": OCT["base"], "r300_42": OCT["r300_s42"], "r0_42": OCT["r0_s42"]}
M.update({k.replace("interns2_abl_", ""): v for k, v in C1127.items()})
M.update(C1128)
M["s43"] = OCT["r300_s43"]          # 回放 300 种子 43 (10 月测的)
M["r0_s43"] = OCT["r0_s43"]
AUDIT = json.loads((R / "slake_no_audit_2026-11-20.json").read_text(encoding="utf-8"))


def r1(x):
    return str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def miss(k):
    return M[k]["freetext"]["abnormal_no_abnormality_reported_pct"]


def disc(k):
    return M[k]["freetext"]["discrimination_pts"]


def normal(k):
    return M[k]["freetext"]["normal_no_abnormality_reported_pct"]


def seq(keys, f=miss):
    return " / ".join(r1(f(k)) for k in keys)


R300, R0 = ["r300_42", "s43", "s44"], ["r0_42", "r0_s43", "r0_s44"]
PQ, D1, D2, NF = ["pqonly", "pqonly_s43"], ["p3dself", "p3dself_s43"], ["p3dcxp", "p3dcxp_s43"], ["p3nofmt"]
BASE = r1(miss("base"))
R1_ALL = D1 + D2
R1_LO, R1_HI = r1(min(miss(k) for k in R1_ALL)), r1(max(miss(k) for k in R1_ALL))
R300_LO, R300_HI = r1(min(miss(k) for k in R300)), r1(max(miss(k) for k in R300))
N_ABN = M["base"]["groups"]["abnormal"]
N_NOR = M["base"]["groups"]["normal"]
N_ALL = M["base"]["n"]
for k in R300 + R0 + PQ + R1_ALL + NF:   # 每个模型都是同一份 3,189 张, 防止拿错文件
    assert M[k]["n"] == N_ALL and M[k]["groups"]["abnormal"] == N_ABN, k
DEGRADED = lambda k: miss(k) > miss("base") + 10  # noqa: E731  只用来数「几个种子退化」, 阈值与 results/README 读法一致: 明显高于原版
N300 = sum(DEGRADED(k) for k in R300)
N0 = sum(DEGRADED(k) for k in R0)


# ---------------------------------------------------------------- 填字工具 (与 make_week7_ppt.py 相同)
def fill(tf, lines):
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


prs = Presentation(str(TEMPLATE))
S = prs.slides
NOTES = []

# ---------------------------------------------------------------- 1 封面
fill(shape(S[0], "details").text_frame, [
    "医学多模态模型的参数高效增量预训练与指令微调",
    "Parameter-Efficient Incremental Pretraining and Instructional Fine-Tuning of Medical Multimodal Models",
    "LOO CHUN QIAN    导师 / Supervisor：徐蕾",
    "第 11 周 / Week 11    2026-12-03",
    "",
    "本周核心进展：第 1–2 周两道关有结果：第一道关过了，第二道关没过，请老师定方向",
    "Main update: weeks 1–2 gates done: gate 1 passed, gate 2 failed; direction needed",
])
NOTES.append("老师好。这次汇报第 9 周定下的第 1 到 2 周计划：补种子，再做三道对照，看两道关。先说结论：第一道关过了，第二道关没过。"
             "按第 9 周定的规则，主线要请老师定方向。下面先讲做了什么，再讲结果，最后请老师选方向。")

# ---------------------------------------------------------------- 2 本周进展 (第 9 周的承诺)
fill_table(table(S[1]), [
    [("补种子：两种短答配方各补到 3 个", "Seeds: 3 per short-answer recipe"),
     ("完成：种子 44 各一个", "Done: one seed-44 run each"),
     (f"回放 300 {N300}/3 退化，回放 0 {N0}/3", f"Replay-300 {N300}/3 degrade, replay-0 {N0}/3")],
    [("三道对照 R0 / R2 / R1", "Three controls R0 / R2 / R1"),
     ("完成：共 18 个模型", "Done: 18 models"),
     ("回放 0 上 8 个、回放 300 上 10 个；3 个待测", "8 on replay-0, 10 on replay-300; 3 pending")],
    [("数据审计与病人隔离", "Data audit and patient isolation"),
     ("完成：三项都查了", "Done: all three checked"),
     ("「没提当没有」不成立；胸片 0 个共同病人", "Audit clear; 0 shared X-ray patients")],
])
fill(shape(S[1], "takeaway").text_frame, [
    "没做：简单回放基线、等算力对照、CheXbert 复核    用卡：苏学长借的 4–7 号卡，世龙的 0–3 号未动",
    "Not done: simple replay, compute-matched control, CheXbert    GPUs: cards 4–7 lent by Su",
])
NOTES.append(f"第 9 周承诺的三件事都做了。第一，补种子：两种短答配方各补了种子 44，现在各有 3 个种子。"
             f"第二，三道对照：先在回放 0 上做了 8 个，因为回放 0 没过第一道关，又在回放 300 上重做了 10 个，其中 7 个已经测完胸片，还差 3 个。"
             f"第三，数据审计：SLAKE 训练集里 {AUDIT['answer_no']} 个答「否」的题，说没有异常的只有 {AUDIT['no_by_kind']['neg_global']} 个，"
             "所以「把没提到当成没有来教」这个解释不成立；三张考卷都查过隔离；胸片训练报告池和测试集，按病人号实数是 0 个共同病人。"
             "还没做的是简单回放基线、等算力对照和 CheXbert 复核。")

# ---------------------------------------------------------------- 3 核心挑战与研究问题
boxes(S[2], [
    ["只练短答后，问答准了，胸片描述里却更常不写异常（漏报）",
     "After short-answer tuning, QA improves but X-ray descriptions omit findings"],
    ["换个种子结果能差很多；还可能只是格式、风格或数据太窄造成的",
     "Results swing across seeds; may be just format, style or narrow data"],
    ["第一道关：3 个种子里至少 2 个退化吗？第二道关：三道对照都消不掉吗？",
     "Gate 1: do ≥2 of 3 seeds degrade? Gate 2: does no control remove it?",
     "两关都过才算「事实退化」，主线成立 / Both must pass for the main line"],
], [
    "第 9 周定的规则：两关都过，主线成立；否则请老师定方向（本页口径不变）",
    "Rule set in week 9: both gates pass → main line holds; otherwise the supervisor decides",
])
NOTES.append("回顾问题：只练短答以后，是非题准了约 10 分，可是让模型描述胸片，它更常一个异常都不写，我们叫漏报。"
             "但这可能只是种子运气，也可能只是格式、说话风格或者数据太单一造成的。所以第 9 周定了两道关：第一道，3 个种子里至少 2 个退化；"
             "第二道，三道便宜的对照都消不掉它。两关都过，才能说是事实退化，主线成立；否则请老师定方向。")

# ---------------------------------------------------------------- 4 文献 (三道对照各自的来源)
fill_table(table(S[3]), [
    [("LLaVA-1.5", "Liu et al., CVPR 2024"),
     ("加「用一个词回答」提示 → R0", "One-word answer prompt → R0"),
     ("通用模型，没测医学描述漏报", "General domain; no medical omission")],
    [("SEFE", "Chen et al., ICML 2025"),
     ("部分遗忘只是风格；改写 → R2", "Some forgetting is style → R2"),
     ("通用持续学习，没测视觉证据", "Continual tuning; no visual evidence")],
    [("Fine-tuning Without Forgetting", "arXiv 2603.14493, 2026"),
     ("混入多样任务数据能缓解 → R1", "Mixing diverse tasks helps → R1"),
     ("没有医学、描述、多种子", "No medicine, descriptions or seeds")],
])
fill(shape(S[3], "takeaway").text_frame, [
    "三道对照各有出处；本周结果：R1 这一条（混描述数据）在我们的设置下确实消掉了漏报",
    "Each control has a source; this week: R1 (mixing descriptions) does remove the omission here",
])
NOTES.append("三道对照都有出处。R0 来自 LLaVA-1.5，它在短答题后加「用一个词回答」，让模型分清什么时候该答短。"
             "R2 来自 SEFE，它认为一部分遗忘只是回答风格变了，把 10% 到 20% 的短答改写成完整的话就能修。"
             "R1 来自 Fine-tuning Without Forgetting，混入多样的任务数据能缓解遗忘。本周的结果是：R1 这一条在我们的设置下确实把漏报消掉了。")

# ---------------------------------------------------------------- 5 方法 (三道对照怎么做)
boxes(S[4], [
    ["R0：训练数据本来每题都带格式提示句，所以新训「去掉提示句」那一边",
     "R0: training data already had the prompt, so we trained the 'removed' side"],
    ["R2：20% 短答（984 条）由原版模型改成一句完整的话，953 条过检查",
     "R2: 20% of answers (984) rewritten as a sentence; 953 passed the check"],
    ["R1 甲：加原版自述 450 条；R1 乙：加胸片报告 490 条（与测试 0 共同病人）",
     "R1a: +450 self-descriptions; R1b: +490 X-ray reports (0 shared patients)"],
], [
    "和第 9 周计划不同：R1 是加样本，多训约 9% 步数；其余设置与回放 300 相同，种子 42、43",
    "Deviation from week 9: R1 added data (~9% more steps); all else as replay-300; seeds 42, 43",
])
NOTES.append("三道对照每次只改一样。R0 要跟老师说明一下：我们发现现有训练数据每道题后面本来就带格式提示句，所以新训的是去掉提示句那一边。"
             "R2 用原版模型把 20% 的短答，一共 984 条，改写成一句完整的话，不加新内容，953 条过了检查。"
             "R1 分两种：甲是原版模型自己描述 450 张训练图，乙是 490 条胸片报告，和测试集没有共同病人。"
             "有一点和第 9 周计划不一样：计划说 R1 总步数不变，实际做成了在原数据上加样本，多训了约 9% 的步数，这一点后面要补。")

# ---------------------------------------------------------------- 6 实验计划
fill_table(table(S[5]), [
    [("问题与对比", "Question / comparison"),
     ("回放 300 配方上，加一样东西，漏报还在不在", "On replay-300, does one change remove omission?")],
    [("数据与划分", "Data / split"),
     ("SLAKE 4,919 条 + PubMedQA 300 条；测试：成人胸片 3,189 张", "SLAKE 4,919 + PubMedQA 300; test: 3,189 adult CXRs")],
    [("指标与控制变量", "Metrics / controls"),
     ("异常片描述没报异常 %（↓）；区分 = 正常片说正常 − 漏报（↑）", "Omission % (↓); discrimination = normal-correct − omission (↑)")],
    [("消融与成功标准", "Ablation / criterion"),
     ("另训只用 PubMedQA 拆原因；判断用关键词规则", "PubMedQA-only run isolates the cause; keyword judge")],
])
fill(shape(S[5], "takeaway").text_frame, [
    f"测试集：异常 {N_ABN:,} 张、正常 {N_NOR} 张；每个漏报比例的 95% 区间约 ±1.5–1.9 点",
    f"Test set: {N_ABN:,} abnormal, {N_NOR} normal; 95% CI of each omission rate is about ±1.5–1.9 pts",
])
NOTES.append(f"实验设计：都在回放 300 这个配方上做，就是 SLAKE 4,919 条短答加 300 条 PubMedQA，因为它三个种子都退化。测试是成人胸片 3,189 张，"
             f"其中异常片 {N_ABN:,} 张、正常片 {N_NOR} 张。主要看漏报：异常片的描述里一个异常都没写的比例，越低越好；"
             "同时看区分：正常片说正常的比例减去漏报，防止模型靠什么都报异常来降低漏报。另外训了一个只用这 300 条 PubMedQA 的模型，拆开看原因。"
             "要说明：判断用的是关键词规则，还没用 CheXbert 复核。")

# ---------------------------------------------------------------- 7 实验结果
t = table(S[6])
set_rows(t, 5, row_h=0.6)
fill_table(t, [
    [("原版（基座）", "Base model"), (f"{BASE}", "omission %"), (f"{r1(disc('base'))}", "discrimination"), ("参照", "reference")],
    [("回放 300 · 3 个种子", "Replay-300, 3 seeds"), (seq(R300), "omission %"), (seq(R300, disc), "discrimination"),
     (f"{N300}/3 退化", f"{N300}/3 degrade")],
    [("回放 0 · 3 个种子", "Replay-0, 3 seeds"), (seq(R0), "omission %"), (seq(R0, disc), "discrimination"),
     (f"{N0}/3 退化", f"{N0}/3 degrade")],
    [("只用 PubMedQA · 2 个种子", "PubMedQA only, 2 seeds"), (seq(PQ), "omission %"), (seq(PQ, disc), "discrimination"),
     ("没有退化", "no degradation")],
    [("回放 300 + R1 甲 / 乙", "Replay-300 + R1a / R1b"), (f"{seq(D1)} | {seq(D2)}", "omission %"),
     (f"{seq(D1, disc)} | {seq(D2, disc)}", "discrimination"), ("4 个都消掉", "removed in all 4")],
])
fill(shape(S[6], "takeaway").text_frame, [
    f"第一道关：回放 300 过（{N300}/3）  第二道关：R1 消掉 → 没过；R0 更差（{seq(NF)}，种子 42）；R2 待测",
    f"Gate 1: passed ({N300}/3)  Gate 2: R1 removes it → failed; R0 worse ({seq(NF)}, seed 42); R2 pending",
])
NOTES.append(f"结果。原版的漏报是 {BASE}%。回放 300 三个种子分别是 {seq(R300)}，三个都退化，第一道关过了。"
             f"回放 0 只有种子 42 退化，另外两个是 {r1(miss('r0_s43'))} 和 {r1(miss('r0_s44'))}，比原版还低，所以回放 0 没过。"
             f"只用 PubMedQA 的两个种子是 {seq(PQ)}，没有退化，所以不是 PubMedQA 单独造成的。"
             f"第二道关：R0 去掉提示句反而更差，{seq(NF)}%，提示句不是原因，种子 43 还在等评测。"
             f"R1 两种、两个种子都把漏报降到 {R1_LO} 到 {R1_HI}%，比原版还低，差距远大于误差。按规则，对照把退化消掉了，所以第二道关没过。"
             f"代价是模型更爱报异常：正常片说正常从 {r1(min(normal(k) for k in R300))} 到 {r1(max(normal(k) for k in R300))}% 降到 {r1(min(normal(k) for k in R1_ALL))} 到 {r1(max(normal(k) for k in R1_ALL))}%，"
             "但区分回到了原版水平。R2 的两个种子胸片还没测。注意第 9 周 PPT 的 25 和 56.5 是 CheXbert 判的，这里是关键词判的，口径不同。")

# ---------------------------------------------------------------- 8 预期成果与阶段结论
boxes(S[7], [
    ["已确认：短答 + PubMedQA 一起练 3/3 退化；单练一样都不稳定退化",
     "SLAKE + PubMedQA: 3/3 seeds degrade; neither alone reliably does"],
    ["已确认：混入描述数据（2 种 × 2 个种子）把漏报降到原版以下",
     "Mixing descriptions (2 kinds × 2 seeds) cuts omission below base"],
    ["未定：是描述本身还是多练 9%；需 CheXbert 复核；R2 待测",
     "Open: descriptions or 9% more steps? CheXbert needed; R2 pending"],
], [
    "所以：现象站得住，但「简单解释排除不掉」这条主线不成立；新方法要换说法或换目标",
    "So: the effect is real, but 'survives simple controls' fails; the method needs a new framing",
])
NOTES.append("阶段结论分三层。站得住的：短答和 PubMedQA 一起练，三个种子都稳定漏报；只练其中一样都没有稳定退化：只练 SLAKE 是 3 个里 1 个，只练 PubMedQA 是 2 个里 0 个。"
             "也站得住的：混入描述数据，两种数据、两个种子，都能把漏报降到原版以下。"
             "还没定的：改善到底是描述数据本身，还是因为多练了约 9% 的步数；关键词判断要用 CheXbert 复核；R2 还没测。"
             "所以现象是真的，但第 9 周说的「排除简单解释后仍然成立」这条主线不成立，原来设计的新方法要换个说法或者换目标。")

# ---------------------------------------------------------------- 9 时间安排
fill_table(table(S[8]), [
    [("第1–8周", "W1–8"), ("基线、35B 三步训练", "Baselines, 3-stage 35B"),
     ("results/ 快照、报告 v1.6", "Snapshots, report v1.6"), ("完成", "Done")],
    [("第9–11周", "W9–11"), ("文献、提案、两道关", "Review, proposal, gates"),
     ("第一关过、第二关没过", "Gate 1 pass, gate 2 fail"), ("完成", "Done")],
    [("第12周", "W12"), ("补测 3 个、等步数 R1", "3 evals, step-matched R1"),
     ("等步数后漏报是否仍降", "Does omission still drop?"), ("待老师定", "Pending decision")],
    [("第13–16周", "W13–16"), ("按老师定的方向做主实验", "Main study per decision"),
     ("毕设表 1–3（12-11 截止）", "Thesis Tables 1–3 (Dec 11)"), ("未开始", "Not started")],
])
fill(shape(S[8], "takeaway").text_frame, [
    "下周承诺（需约 4 张卡约 6 小时）：补测 R0 种子 43 与 R2 两个；训 R1 等步数版；问队友后做 CheXbert",
    "Next week (~6 h on 4 GPUs): 3 pending evals; step-matched R1; CheXbert after asking teammates",
])
NOTES.append("时间安排：前 11 周的内容都做完了。下周想先做三件事：补测 R0 种子 43 和 R2 两个种子的胸片；训一个总步数和对照组相同的 R1，"
             "看漏报还降不降，这决定改善是不是描述数据本身带来的；问过队友后用 CheXbert 复核。大约要 4 张卡 6 小时。"
             "毕设表 1 到 3 在 12 月 11 日截止，题目要等方向定了再和老师确认。")

# ---------------------------------------------------------------- 10 讨论与决策
boxes(S[9], [
    ["A：改成「诊断 + 简单修复」（混描述就能修）",
     "A: diagnosis + simple fix (description mixing)",
     "B：保留新方法，须在同等步数下胜过 R1",
     "B: keep the new method; it must beat R1 at equal steps"],
    ["C：先补等步数 R1 和 CheXbert，一周后再定（我倾向 C）",
     "C: decide in a week, after step-matched R1 and CheXbert"],
    ["会议决策：（会后填写）", "Decision: (to fill after the meeting)", "执行事项：（会后填写） / Action: (after the meeting)"],
], [
    "每周材料：汇报 PPT、实验日志（results/README.md）、代码（GitHub AugustLoo/MedLoRA）",
    "Weekly package: slides, experiment log (results/README.md), code (GitHub AugustLoo/MedLoRA)",
])
NOTES.append("想请老师定方向，有三个选择。A：把题目改成诊断加简单修复，讲清楚哪种混合配方会让描述漏报、原因是什么、混一点描述数据就能修；"
             "这条证据最扎实，但新意比原计划小。B：保留原来设计的新方法，但它必须在同样步数、同样数据下比 R1 更好，否则审稿人会说直接混描述就够了。"
             "C：先补等步数的 R1 和 CheXbert 复核，一周后再定。我倾向先做 C，因为它决定 A 的结论能不能写死。"
             "另外想确认两件事：R0 是去掉提示句，和第 9 周说的加提示句方向相反，这样写进论文可以吗；评测结果以谁的为准。")

for slide, note in zip(S, NOTES):
    slide.notes_slide.notes_text_frame.text = note

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
TITLES = ["封面", "本周进展", "核心挑战与研究问题", "现有文献与研究空白", "拟议方法与独特贡献",
          "实验计划", "实验结果", "预期成果与阶段结论", "时间安排", "讨论与决策"]
script = ["# 第 11 周汇报讲稿（2026-12-03）", "", "与 PPT 备注相同；每页讲一段，大白话。", ""]
for i, (title, note) in enumerate(zip(TITLES, NOTES), start=1):
    script += [f"## {i}. {title}", "", note, ""]
OUT.with_name("Topic6_WeeklyReport11_Script.md").write_text("\n".join(script), encoding="utf-8")
print("written", OUT, "| r300 degraded", N300, "| r0 degraded", N0, "| R1", R1_LO, "-", R1_HI)
