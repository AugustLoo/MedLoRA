"""把 docs/Topic6_WeeklyReport.html 的内容与视觉做成 PPT: docs/Topic6_WeeklyReport.pptx (16:9)。
先跑 scripts/make_ppt_charts.py 生成图表。用法: python scripts/make_ppt.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[1]
CH = REPO / "docs" / "ppt" / "charts"
FIG = REPO / "docs" / "figures"
OUT = REPO / "docs" / "Topic6_WeeklyReport.pptx"

def rgb(h): return RGBColor.from_string(h.lstrip("#"))
PAPER, DEEP, CARD = rgb("F7F2E7"), rgb("EFE7D3"), rgb("FDFAF2")
NAVY, SOFT, TAUPE, RULE, SHADOW = rgb("14263B"), rgb("3D4E63"), rgb("8C8577"), rgb("D8CFBA"), rgb("D9D1BE")
RED, BLUE, ORANGE, GREEN, GOLDC = rgb("C0392B"), rgb("0E67B5"), rgb("C25E28"), rgb("1F8A70"), rgb("B8860B")
F_HEAD, F_BODY, F_NUM = "Noto Serif SC", "Noto Sans SC", "Consolas"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def new_slide(deep=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid(); bg.fore_color.rgb = DEEP if deep else PAPER
    return s


def rect(s, x, y, w, h, fill, line=None, lw=0):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None:
        r.line.fill.background()
    else:
        r.line.color.rgb = line; r.line.width = Pt(lw)
    r.shadow.inherit = False
    return r


def card(s, x, y, w, h, dashed=False):
    if not dashed:
        rect(s, x + Inches(0.07), y + Inches(0.07), w, h, SHADOW)
    r = rect(s, x, y, w, h, CARD, NAVY, 1.75)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        r.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return r


def text(s, x, y, w, h, runs, size=14, font=F_BODY, color=SOFT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.25, para_gap=4):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, opts) tuples."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.line_spacing = spacing; para.space_after = Pt(para_gap)
        for seg in (p if isinstance(p, list) else [(p, {})]):
            t, o = seg if isinstance(seg, tuple) else (seg, {})
            r = para.add_run(); r.text = t
            f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def kicker(s, x, y, t, w=Inches(8)):
    text(s, x, y, w, Inches(0.3), t.upper(), size=10, font=F_NUM, color=TAUPE)


def h2(s, x, y, t, w=Inches(9), size=30):
    text(s, x, y, w, Inches(0.8), t, size=size, font=F_HEAD, color=NAVY, bold=True, spacing=1.1)
    rect(s, x, y + Inches(0.78), Inches(0.66), Inches(0.05), NAVY)


def en(t, size=None, color=TAUPE):
    o = {"font": F_NUM, "color": color}
    if size: o["size"] = size
    return (t, o)


def picture(s, path, x, y, w=None, h=None, frame=True):
    if frame:
        pic = s.shapes.add_picture(str(path), x, y, w, h)
        card(s, x - Inches(0.12), y - Inches(0.12), pic.width + Inches(0.24), pic.height + Inches(0.24))
        pic = s.shapes.add_picture(str(path), x, y, w, h)  # re-add on top of the card
        return pic
    return s.shapes.add_picture(str(path), x, y, w, h)


def footer(s, n):
    text(s, Inches(0.6), H - Inches(0.45), Inches(8), Inches(0.3), "MEDLORA · TOPIC 6 · 第 2 周进展汇报 · 2026-09-17", size=9, font=F_NUM, color=TAUPE)
    text(s, W - Inches(1.4), H - Inches(0.45), Inches(0.8), Inches(0.3), f"{n:02d}", size=9, font=F_NUM, color=TAUPE, align=PP_ALIGN.RIGHT)


# ---------------- 1. cover ----------------
s = new_slide()
kicker(s, Inches(1.0), Inches(1.15), "Topic 6 · Task 1.3 · 第 2 周进展汇报 · 2026-09-17")
text(s, Inches(1.0), Inches(1.5), Inches(11), Inches(2.4),
     [[("SFT 涨了 22 分，", {"color": NAVY}), ("CPT", {"color": RED}), (" 没有增益", {"color": NAVY})]],
     size=56, font=F_HEAD, bold=True, spacing=1.15)
text(s, Inches(1.0), Inches(3.55), Inches(8.6), Inches(0.9),
     "医学视觉语言模型的参数高效增量预训练与指令微调。两周，四组实验，三张固定的评估表，每一个数字都能在仓库里复现。", size=16, color=SOFT)
text(s, Inches(1.0), Inches(4.4), Inches(10), Inches(0.3), "Qwen2.5-VL-3B-Instruct · QLoRA r16 · Kaggle T4 · results as of 2026-09-16", size=10.5, font=F_NUM, color=TAUPE)
for i, (v, frm, lab, col) in enumerate([("89.2", "← 67.3", "SLAKE closed accuracy，SFT 之后（实验 A）", ORANGE),
                                        ("82.2", "← 46.7", "SLAKE open recall，同一个 adapter", ORANGE),
                                        ("83.9", "= 83.9", "TextVQA，通用能力没有可测的退化", NAVY)]):
    x = Inches(1.0 + i * 3.4)
    text(s, x, Inches(5.0), Inches(3.2), Inches(0.7), [[(v, {"font": F_NUM, "size": 34, "bold": True, "color": col}), ("  " + frm, {"font": F_NUM, "size": 13, "color": TAUPE})]])
    rect(s, x, Inches(5.75), Inches(1.3), Inches(0.06), col)
    text(s, x, Inches(5.88), Inches(3.0), Inches(0.6), lab, size=11, color=SOFT)
text(s, 0, H - Inches(0.55), W, Inches(0.3), "SCROLL ▼".replace("SCROLL ▼", "MEDLORA · github.com/AugustLoo/MedLoRA"), size=9, font=F_NUM, color=TAUPE, align=PP_ALIGN.CENTER)

# ---------------- 2. pipeline ----------------
s = new_slide(deep=True)
kicker(s, Inches(0.8), Inches(0.55), "01 · 课题与流程")
h2(s, Inches(0.8), Inches(0.85), "三个阶段，一套评估")
text(s, Inches(0.8), Inches(1.85), Inches(11.6), Inches(1.0),
     [[("课题任务 1.3：对一个 2B 到 7B 的开源视觉语言模型做增量预训练、LoRA 指令微调和对齐。基座 ", {}), en("Qwen2.5-VL-3B-Instruct", color=SOFT), ("，框架 ", {}), en("LLaMA-Factory", color=SOFT),
       ("，全部在 Kaggle 的 T4 上跑。每个阶段产出一个 LoRA adapter，后一阶段接着前一阶段继续训练；视觉塔全程冻结，所以每一处变化都能归因到语言模型一侧。", {})]], size=14)
cards = [("BASE", NAVY, "基座模型", "出厂状态的 3B 模型，直接做题，作为起点。", "Qwen2.5-VL-3B-Instruct\n4-bit NF4 · vision tower frozen", False),
         ("STAGE 1 · CPT", GREEN, "持续预训练", "让模型自己读材料，没有题目也没有答案，只学下一个词是什么。", "B1: 10k PubMed abstracts (text)\nB2: 3.5k IU X-Ray image → report\nLoRA r16 · lr 5e-5 · 1 epoch", False),
         ("STAGE 2 · SFT", ORANGE, "指令微调", "做题对答案：一张图、一个问题、一个标准答案，答错就往标准答案改。", "SLAKE train 4,919 QA\nsame prompts as eval\nLoRA r16 · lr 1e-4 · 3 epochs", False),
         ("STAGE 3 · ALIGNMENT", TAUPE, "对齐（计划中）", "纠正回答习惯：该说「不确定」时不要硬答「是」。", "yes / no / maybe calibration\nlabel-rebalanced SFT, then DPO", True)]
cw, gap, x0, y0, chh = Inches(2.75), Inches(0.2), Inches(0.8), Inches(3.05), Inches(3.2)
for i, (tag, col, title, body, enl, dashed) in enumerate(cards):
    x = x0 + i * (cw + gap)
    card(s, x, y0, cw, chh, dashed=dashed)
    tg = rect(s, x + Inches(0.2), y0 + Inches(0.2), Inches(1.55), Inches(0.28), col)
    text(s, x + Inches(0.2), y0 + Inches(0.2), Inches(1.55), Inches(0.28), tag, size=8.5, font=F_NUM, color=CARD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + Inches(0.2), y0 + Inches(0.58), cw - Inches(0.4), Inches(0.45), title, size=17, font=F_HEAD, color=NAVY, bold=True)
    text(s, x + Inches(0.2), y0 + Inches(1.05), cw - Inches(0.4), Inches(1.1), body, size=12, color=SOFT)
    text(s, x + Inches(0.2), y0 + Inches(2.15), cw - Inches(0.4), Inches(1.0), enl, size=9.5, font=F_NUM, color=TAUPE, spacing=1.2)
text(s, Inches(0.8), Inches(6.5), Inches(11.6), Inches(0.4), [[("每个阶段都用同一套三张表评估：", {}), en("SLAKE test · TextVQA-300 · PubMedQA labeled", color=SOFT), ("。实验 A 跳过 Stage 1，B1 与 B2 只在 Stage 1 的语料上不同。", {})]], size=11.5, color=TAUPE)
footer(s, 2)

# ---------------- 3. eval tables ----------------
s = new_slide()
kicker(s, Inches(0.8), Inches(0.55), "02 · 怎么评")
h2(s, Inches(0.8), Inches(0.85), "三张表，固定不动")
text(s, Inches(0.8), Inches(1.85), Inches(11.6), Inches(0.8), "提示词只写一次，训练数据生成和评估用同一套；解码用贪心，最多 32 个词，图片上限 512 像素。这样每一轮的分数才能直接相减。", size=14)
rows = [("表", "数据", "回答的问题", "指标"),
        ("医学能力", "SLAKE test · 1,061 questions\n(416 closed / 645 open)", "医学看图问答涨了多少", "closed acc · open EM / recall / F1\nby modality"),
        ("通用保持", "TextVQA validation\nfixed 300, seed 42", "学了医学之后，通用能力掉了多少", "VQA accuracy"),
        ("可靠性", "PubMedQA pqa_labeled · 1,000", "证据不够时敢不敢说 maybe", "accuracy · macro-F1\nyes / no / maybe distribution")]
tx, ty, tw = Inches(0.8), Inches(2.85), Inches(11.7)
colw = [Inches(1.6), Inches(3.6), Inches(3.2), Inches(3.3)]
rh = [Inches(0.45), Inches(0.95), Inches(0.95), Inches(0.95)]
rect(s, tx + Inches(0.07), ty + Inches(0.07), tw, sum(rh, Emu(0)), SHADOW)
rect(s, tx, ty, tw, sum(rh, Emu(0)), CARD, NAVY, 1.75)
y = ty
for ri, row in enumerate(rows):
    if ri == 0:
        rect(s, tx, y, tw, rh[0], DEEP)
        rect(s, tx, y + rh[0] - Inches(0.02), tw, Inches(0.02), NAVY)
    elif ri < len(rows) - 1:
        rect(s, tx, y + rh[ri] - Emu(6350), tw, Emu(6350), RULE)
    x = tx
    for ci, cell in enumerate(row):
        if ri == 0:
            text(s, x + Inches(0.15), y, colw[ci] - Inches(0.3), rh[0], cell.upper(), size=9.5, font=F_NUM, color=SOFT, anchor=MSO_ANCHOR.MIDDLE)
        else:
            mono = ci in (1, 3)
            text(s, x + Inches(0.15), y + Inches(0.08), colw[ci] - Inches(0.3), rh[ri] - Inches(0.1), cell,
                 size=11 if mono else 12.5, font=F_NUM if mono else F_BODY, color=SOFT if mono else NAVY, bold=(ci == 0), spacing=1.2)
        x += colw[ci]
    y += rh[ri]
text(s, Inches(0.8), Inches(6.4), Inches(11.6), Inches(0.5), [[("红线：", {}), en("SLAKE test", color=SOFT), (" 与 ", {}), en("PubMedQA labeled", color=SOFT), (" 只评估、永不训练；切分文件固定入库；受控数据不进 git。", {})]], size=11.5, color=TAUPE)
footer(s, 3)

# ---------------- 4-6. main results (3 slides, chart + step cards) ----------------
def result_slide(n, deep, chart, title, steps):
    s = new_slide(deep=deep)
    kicker(s, Inches(0.8), Inches(0.55), "03 · 主结果")
    h2(s, Inches(0.8), Inches(0.85), title, size=28)
    picture(s, CH / chart, Inches(0.92), Inches(2.25), w=Inches(7.6))
    x, y = Inches(8.95), Inches(1.95)
    for (num, t, body) in steps:
        hgt = Inches(1.05 + 0.24 * (len(body) // 34))
        card(s, x, y, Inches(3.75), hgt)
        text(s, x + Inches(0.2), y + Inches(0.12), Inches(3.4), Inches(0.25), num, size=8.5, font=F_NUM, color=TAUPE)
        text(s, x + Inches(0.2), y + Inches(0.34), Inches(3.4), Inches(0.35), t, size=13.5, font=F_HEAD, color=NAVY, bold=True)
        text(s, x + Inches(0.2), y + Inches(0.68), Inches(3.4), hgt - Inches(0.7), body, size=10.5, color=SOFT, spacing=1.3)
        y += hgt + Inches(0.18)
    footer(s, n)

result_slide(4, True, "main_a.png", "起点，以及只做 SFT 涨了多少",
             [("01 / 05", "起点：基座直接做题", "SLAKE 封闭题 67.3，开放题 recall 46.7。X 光片最好、MRI 最差，说明基座见过的胸片远多于 MRI。PubMedQA 明显偏 yes。"),
              ("02 / 05", "A：只做 SFT，涨 22 分", "4,919 道 SLAKE 题做三遍，4 小时 28 分。封闭题 67.3 → 89.2，开放题 recall 46.7 → 82.2。逐题看，修正 345 题，新错 30 题。")])
result_slide(5, True, "main_all.png", "加上 CPT 之后，曲线没有再往上走",
             [("03 / 05", "B1：先读一万篇医学摘要，再 SFT", "SLAKE 与 A 的差异全部在 ±0.6 以内，逐题互换 15 对 14，是随机波动。文本 CPT 没有跨过模态，只帮到了自己的领域：PubMedQA 70.6 → 72.8。"),
              ("04 / 05", "B2：先看三千多张胸片配报告，再 SFT", "封闭题 88.7 与 B1 持平，开放题略低 −0.6，逐题修正 11 题、新错 18 题。图文 CPT 同样没有增益。"),
              ("05 / 05", "通用能力：一分没掉，但探针太钝", "TextVQA 四个模型都是 84 上下，300 题里 81 题只差大小写。短答题和 SFT 输出格式太像，测不出退化，实验 C 之前要换基准。")])

# ---------------- 7. by type ----------------
s = new_slide()
kicker(s, Inches(0.8), Inches(0.55), "04 · 拆开看")
h2(s, Inches(0.8), Inches(0.85), "涨的是词表对齐，没涨的是真识别", size=28)
picture(s, CH / "bytype.png", Inches(0.92), Inches(2.0), w=Inches(7.45))
x, y = Inches(8.95), Inches(1.9)
for num, t, body in [("01 / 03", "基座会看，但不会答", "器官、知识图谱、颜色、平面这些题型，基座只有 22 到 40 分。不是看不懂，是答法不对。SFT 把它们一口气拉到 85 到 100，这是词表和格式的对齐。"),
                     ("02 / 03", "真正的识别题只到四成和六成", "Abnormality open 41.5，Position open 57.7。剩余错例是左右、上下弄反和病灶类别混淆，是后面要攻的地方。"),
                     ("03 / 03", "B2 在病灶题上反而掉了", "Abnormality open 41.5 → 31.7，41 题少对 4 题，是四组实验里唯一超出噪声的下降。其余题型 B1、B2 都贴着 A。")]:
    hgt = Inches(1.5)
    card(s, x, y, Inches(3.85), hgt)
    text(s, x + Inches(0.2), y + Inches(0.12), Inches(3.5), Inches(0.25), num, size=8.5, font=F_NUM, color=TAUPE)
    text(s, x + Inches(0.2), y + Inches(0.34), Inches(3.5), Inches(0.35), t, size=13.5, font=F_HEAD, color=NAVY, bold=True)
    text(s, x + Inches(0.2), y + Inches(0.68), Inches(3.5), Inches(0.8), body, size=10.5, color=SOFT, spacing=1.3)
    y += hgt + Inches(0.18)
footer(s, 6)

# ---------------- 8. examples ----------------
s = new_slide(deep=True)
kicker(s, Inches(0.8), Inches(0.55), "05 · 看几道题")
h2(s, Inches(0.8), Inches(0.85), "基座答错、微调后全对的三道题", size=28)
text(s, Inches(0.8), Inches(1.8), Inches(11.6), Inches(0.5), "三道 CT 题，分别是病灶、位置、器官。基座的错各有各的样子，三个微调模型给出完全相同的正确答案。", size=13.5)
exs = [("xmlab102.jpg", "CT · ABNORMALITY · OPEN", "What diseases are included in the picture?", "Lung Cancer", "Pneumonia ✗", "Lung Cancer ✓"),
       ("xmlab103.jpg", "CT · POSITION · OPEN", "Where is/are the abnormality located?", "Left Lung, Right", "Right lung ✗", "Left Lung, Right ✓"),
       ("xmlab105.jpg", "CT · ORGAN · CLOSED", "Does the picture contain heart?", "No", "Yes. ✗", "No ✓")]
cw, x0, y0 = Inches(3.7), Inches(0.8), Inches(2.45)
for i, (img, tag, q, gold, base, ft) in enumerate(exs):
    x = x0 + i * (cw + Inches(0.25))
    card(s, x, y0, cw, Inches(4.3))
    s.shapes.add_picture(str(FIG / "slake" / img), x + Inches(0.2), y0 + Inches(0.2), height=Inches(1.9))
    text(s, x + Inches(0.2), y0 + Inches(2.15), cw - Inches(0.4), Inches(0.25), tag, size=8, font=F_NUM, color=TAUPE)
    text(s, x + Inches(0.2), y0 + Inches(2.4), cw - Inches(0.4), Inches(0.55), q, size=12, color=NAVY, bold=True, spacing=1.2)
    yy = y0 + Inches(2.98)
    for k, v, col in [("Gold", gold, NAVY), ("Base", base, RED), ("A / B1 / B2", ft, GREEN)]:
        rect(s, x + Inches(0.2), yy, cw - Inches(0.4), Emu(6350), RULE)
        text(s, x + Inches(0.2), yy + Inches(0.03), Inches(1.3), Inches(0.4), k, size=10, font=F_NUM, color=TAUPE)
        text(s, x + Inches(1.4), yy + Inches(0.03), cw - Inches(1.6), Inches(0.4), v, size=10, font=F_NUM, color=col, align=PP_ALIGN.RIGHT)
        yy += Inches(0.42)
text(s, Inches(0.8), Inches(6.84), Inches(8), Inches(0.3), "SLAKE test, qid 11941 / 11953 / 11961 · images CC BY 4.0", size=9, font=F_NUM, color=TAUPE)
footer(s, 7)

# ---------------- 9. CPT learned ----------------
s = new_slide()
kicker(s, Inches(0.8), Inches(0.55), "06 · CPT 到底学到了什么")
h2(s, Inches(0.8), Inches(0.85), "学会了报告怎么写，没学会片子怎么看", size=28)
text(s, Inches(0.8), Inches(1.85), Inches(5.4), Inches(4.6),
     ["B2 的 CPT 阶段结束后，先不做 SFT，直接让它给三张没见过的胸片写报告。写出来的东西很像放射科报告，但两张有异常的片子都被写成了正常。",
      [("眼睛没有参与学习", {"color": NAVY, "bold": True}), ("：视觉塔是冻结的，三千张胸片交给大脑的特征和训练前一模一样，大脑只学到了「拿到这种特征之后通常写什么话」。", {})],
      [("材料本身有偏", {"color": NAVY, "bold": True}), ("：IU X-Ray 训练集 36% 是正常报告，异常报告也大多用「no effusion」这种否定句写，caption 式损失奖励的是高频模板。CPT 损失 2.17 → 1.06，目标学会了，学会的是模板。这个「一切正常」的先验带进 SFT，就是病灶题掉 4 题的来源。", {})]],
     size=12.5, spacing=1.35, para_gap=8)
s.shapes.add_picture(str(FIG / "iu" / "1015_IM-0001-1001.jpg"), Inches(6.6), Inches(1.95), height=Inches(3.1))
card(s, Inches(6.48), Inches(1.83), Inches(2.75), Inches(3.34))
s.shapes.add_picture(str(FIG / "iu" / "1015_IM-0001-1001.jpg"), Inches(6.6), Inches(1.95), height=Inches(3.1))
text(s, Inches(6.5), Inches(5.25), Inches(2.8), Inches(0.3), "IU X-Ray held-out · 1015_IM-0001-1001", size=8, font=F_NUM, color=TAUPE)
card(s, Inches(9.55), Inches(1.83), Inches(3.2), Inches(2.85))
text(s, Inches(9.7), Inches(1.93), Inches(2.9), Inches(0.25), "REFERENCE REPORT (RADIOLOGIST)", size=8, font=F_NUM, color=TAUPE)
text(s, Inches(9.7), Inches(2.18), Inches(2.9), Inches(2.5),
     [[("Findings: ", {}), ("Streaky and patchy bibasilar opacities", {"color": GREEN, "bold": True}), (", triangular density projected over the heart on the lateral view. No definite pleural effusion seen, no typical findings of pulmonary edema. Impression: ", {}), ("Bibasilar opacities, right greater than left, consolidation and atelectasis", {"color": GREEN, "bold": True}), (".", {})]],
     size=9, font=F_NUM, color=NAVY, spacing=1.25)
card(s, Inches(9.55), Inches(4.85), Inches(3.2), Inches(1.75))
text(s, Inches(9.7), Inches(4.95), Inches(2.9), Inches(0.25), "GENERATED BY B2 CPT ADAPTER, BEFORE SFT", size=8, font=F_NUM, color=TAUPE)
text(s, Inches(9.7), Inches(5.2), Inches(2.9), Inches(1.4),
     [[("Findings: ", {}), ("The heart and lungs are grossly unremarkable.", {"color": RED, "bold": True}), (" No pleural effusion or pneumothorax. Impression: ", {}), ("No acute cardiopulmonary abnormality.", {"color": RED, "bold": True})]],
     size=9, font=F_NUM, color=SOFT, spacing=1.25)
footer(s, 8)

# ---------------- 10. reliability ----------------
s = new_slide(deep=True)
kicker(s, Inches(0.8), Inches(0.55), "07 · 可靠性")
h2(s, Inches(0.8), Inches(0.85), "越训越不敢说「不确定」", size=28)
text(s, Inches(0.8), Inches(1.85), Inches(4.6), Inches(4.8),
     ["PubMedQA 一千题里真实答案有 110 个 maybe。基座答了 158 个，A 只剩 59，B1 剩 44。准确率在涨，是因为短答题训练把犹豫训掉了，不是因为判断变准了。",
      "该答 no 却答成 yes 的题：基座 85、A 106、B1 92、B2 99，四轮没有一轮改善；该答 maybe 的 110 题，基座答对 17 题，三个微调模型都只答对 8 到 9 题。",
      "这是第三阶段「对齐」要打的靶子：先试成本最低的按标签重新平衡的 SFT，不行再上 DPO。",
      [("偏置是 SFT 带进来的，不是 CPT。", {"color": NAVY, "bold": True}), ("只做 CPT 不做 SFT 的 B2 adapter 答 yes / no / maybe 598 / 288 / 114，五个模型里最接近真实分布；一做短答式 SFT，maybe 就塌下去。对齐要改的是 SFT 的数据配比。", {})]], size=12.5, spacing=1.35, para_gap=8)
picture(s, CH / "pubmedqa.png", Inches(5.85), Inches(2.1), w=Inches(6.9))
footer(s, 9)

# ---------------- 11. curves ----------------
s = new_slide()
kicker(s, Inches(0.8), Inches(0.55), "08 · 训练过程")
h2(s, Inches(0.8), Inches(0.85), "五次训练，没有一次出问题", size=28)
text(s, Inches(0.8), Inches(1.85), Inches(11.6), Inches(0.6), "训练损失单调下降，验证损失到第三轮还在降，没有 NaN。A 的验证损失 0.225 → 0.145，说明三轮没有过拟合。", size=13.5)
picture(s, CH / "curves.png", Inches(0.92), Inches(2.85), w=Inches(11.5))
text(s, Inches(0.8), Inches(6.35), Inches(11.6), Inches(0.5), "Wall time on one T4: A 4 h 28 min · B1 CPT 2 h 39 min + SFT 4 h 07 min · B2 CPT 1 h 19 min + SFT 4 h 38 min · each evaluation ≈ 1.5 h", size=10, font=F_NUM, color=TAUPE)
footer(s, 10)

# ---------------- 12. conclusions ----------------
s = new_slide(deep=True)
kicker(s, Inches(0.8), Inches(0.55), "09 · 结论")
h2(s, Inches(0.8), Inches(0.85), "主结果是 A，B1 和 B2 是对照证据", size=28)
text(s, Inches(0.8), Inches(1.85), Inches(11.6), Inches(0.9), "在 3B 模型、冻结视觉塔、几千条数据的条件下，CPT 阶段对医学 VQA 的主指标没有贡献。这是有对照的结论，不是失败的实验，报告里作为「CPT 何时无效」的一章保留。", size=14)
concl = [("+21.9", "SLAKE closed", ORANGE, "SFT 有效，而且主要是词表对齐", "真识别题只到四成和六成，是下一阶段的目标。"),
         ("±0.6", "B1 vs A", GREEN, "文本 CPT 只帮文本任务", "PubMedQA +2.2，SLAKE 不动。内容和考题对不上就没有迁移。"),
         ("−9.8", "Abnormality open", GOLDC, "图文 CPT 学到模板，伤了病灶题", "要让 CPT 起作用只有两条路：解冻视觉塔，或换异常均衡的语料。")]
cw, x0, y0 = Inches(3.75), Inches(0.8), Inches(3.1)
for i, (big, lab, col, t, body) in enumerate(concl):
    x = x0 + i * (cw + Inches(0.22))
    card(s, x, y0, cw, Inches(3.0))
    text(s, x + Inches(0.25), y0 + Inches(0.2), Inches(3.3), Inches(0.7), [[(big, {"font": F_NUM, "size": 32, "bold": True, "color": col}), ("  " + lab, {"font": F_NUM, "size": 11, "color": TAUPE})]])
    text(s, x + Inches(0.25), y0 + Inches(1.05), Inches(3.3), Inches(0.8), t, size=15, font=F_HEAD, color=NAVY, bold=True, spacing=1.2)
    text(s, x + Inches(0.25), y0 + Inches(1.85), Inches(3.3), Inches(1.0), body, size=11.5, color=SOFT, spacing=1.3)
footer(s, 11)

prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
