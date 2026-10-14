"""第 5 周周报 PPT (5 页, 中文 + 简短英文): docs/weekly/week5/Topic6_WeeklyReport5.pptx
数字全部来自 results/interns2_6ds_2026-10-10.json、results/interns2_cxp_2026-10-13.json 与 outputs/eval/chexbert_interns2_6ds.json 的汇总。
用法: python scripts/make_week5_ppt.py
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "weekly" / "week5" / "Topic6_WeeklyReport5.pptx"


def rgb(h):
    return RGBColor.from_string(h)


NAVY, NAVY2, PAPER, CARD, DEEP = rgb("14263B"), rgb("1F3550"), rgb("F7F2E7"), rgb("FDFAF2"), rgb("EFE7D3")
INK, SOFT, TAUPE, RULE, WHITE, PALE = rgb("14263B"), rgb("3D4E63"), rgb("8C8577"), rgb("D8CFBA"), rgb("FFFFFF"), rgb("C9D3E0")
BLUE, GOLD, GREEN, PURPLE, RED = rgb("0E67B5"), rgb("B8860B"), rgb("1F8A70"), rgb("8E44AD"), rgb("C0392B")
CN, NUM = "Microsoft YaHei", "Arial"

# ---------------------------------------------------------------- 数字 (从快照读, 不手抄)
cxp = json.loads((REPO / "results/interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
judge = json.loads((REPO / "outputs/eval/chexbert_interns2_6ds.json").read_text(encoding="utf-8"))
MODELS = [("原版", "Base", "base", BLUE), ("S2-2datasets", "deployed", "S2-2datasets", GOLD),
          ("S2-6datasets", "6 datasets", "S2-6datasets", GREEN), ("S2-CheXpertPlus", "X-ray reports", "S2-CheXpertPlus", PURPLE)]
HELD = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])


def maybe_correct(tag):
    rows = [json.loads(x) for x in open(REPO / f"outputs/eval/pubmedqa_{tag}_preds.jsonl", encoding="utf-8")]
    return sum(r["gold"] == r["pred"] == "maybe" for r in rows if r["pubid"] in HELD)


V = {}
for _, _, k, _ in MODELS:
    m = cxp[k]
    V[k] = {"slake": m["slake_closed"], "maybe": maybe_correct(m["tag"]), "path": m["pathvqa_closed"], "medqa": m["medqa"],
            "omit": m["chexpert"]["freetext"]["abnormal_no_abnormality_reported_pct"],
            "disc": m["chexpert"]["freetext"]["discrimination_pts"],
            "falarm": round(100 - m["chexpert"]["closed"]["specificity"], 1)}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(bg):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def text(s, x, y, w, h, runs, size=16, color=INK, bold=False, font=CN, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=0):
    """runs: 字符串, 或 [(文字, {size,color,bold,font}) ...] 的段落列表 (每个元素一段)。"""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [(runs, {})]
    for i, (t, o) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(o.get("space", space))
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.color.rgb = o.get("color", color)
        f.name = o.get("font", font)
        set_ea(r)
    return tb


def set_ea(r):
    """东亚字符一律用雅黑 (否则 Arial 段落里的中文会回落成宋体)。"""
    rpr = r._r.get_or_add_rPr()
    rpr.set("lang", "zh-CN")
    if rpr.find(qn("a:ea")) is None:
        rpr.append(rpr.makeelement(qn("a:ea"), {"typeface": CN}))


def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line
        b.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = radius
    b.shadow.inherit = False
    return b


def title(s, cn, en, dark=False):
    text(s, 0.6, 0.45, 12, 0.7, [(cn, {"size": 34, "bold": True, "color": WHITE if dark else INK})])
    text(s, 0.6, 1.15, 12, 0.4, [(en, {"size": 15, "color": PALE if dark else TAUPE, "font": NUM})])


def circle_num(s, x, y, n, color):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.5), Inches(0.5))
    c.fill.solid()
    c.fill.fore_color.rgb = color
    c.line.fill.background()
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(18), True, WHITE, NUM
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


# ================================================================ 1 封面
s = slide(NAVY)
text(s, 0.8, 0.7, 11.5, 0.4, [("Topic 6 · Task 1.3 · 第 5 周汇报 · Week 5 · 2026-10-15", {"size": 14, "color": rgb("D9B45A"), "font": NUM})])
text(s, 0.8, 1.5, 11.5, 1.0, [("两个新模型训练与评测完成", {"size": 44, "bold": True, "color": WHITE})])
text(s, 0.8, 2.55, 11.5, 0.5, [("Two new 35B models trained and evaluated: S2-6datasets & S2-CheXpertPlus",
                                 {"size": 18, "color": PALE, "font": NUM})])
stats = [(f"{V['S2-2datasets']['omit']:.1f}% → {V['S2-6datasets']['omit']:.1f}%", "成人胸片描述漏报", "Missed findings in descriptions", GREEN),
         (f"{V['S2-2datasets']['path']:.1f} → {V['S2-6datasets']['path']:.1f}", "病理图问答（PathVQA）", "Pathology VQA accuracy", GREEN),
         (f"{judge['agreement_with_keyword_judge_pct']:.1f}%", "CheXbert 与关键词判定一致", "Judge agreement (CheXbert)", rgb("D9B45A"))]
for i, (big, cn, en, col) in enumerate(stats):
    x = 0.8 + i * 4.0
    box(s, x, 3.75, 3.7, 2.2, NAVY2)
    text(s, x + 0.35, 4.0, 3.2, 0.9, [(big, {"size": 28, "bold": True, "color": col, "font": NUM})])
    text(s, x + 0.35, 4.95, 3.1, 0.9, [(cn, {"size": 16, "color": WHITE, "space": 2}), (en, {"size": 12, "color": PALE, "font": NUM})])
text(s, 0.8, 6.65, 11.5, 0.4, [("Chunqian Loo · Intern-S2-Preview 35B · LoRA SFT · 单次训练 single seed", {"size": 12, "color": PALE, "font": NUM})])
notes(s, "这周把老师要求的两个新模型都训练完、评测完了。最重要的结果：描述胸片时把病说成正常的比例，从现在部署的模型的 68.5% 降到 13.8%；"
         "病理图问答从 76.8 涨到 92.7。漏报的判断还用 CheXbert 交叉核对过，和原来的关键词判断 95.6% 一致。")

# ================================================================ 2 本周工作
s = slide(PAPER)
title(s, "本周完成的工作", "What I did this week")
cards = [("S2-6datasets", "六个数据集的训练部分，共 15,953 条；自动去除与测试题重叠的样本（VQA-RAD 去掉 1,059 题）",
          "6 datasets, train splits only, test overlap removed", GREEN),
         ("S2-CheXpertPlus", "1 万对胸片 → 报告；数据在服务器内网传输，1 万个文件逐个核对无误",
          "10k X-ray → report pairs, every file verified", PURPLE),
         ("评测", "四项基础测试 + 四个外部数据集 + 3,189 张成人胸片；用 CheXbert 复核漏报判断",
          "Full evaluation, missed findings re-checked with CheXbert", BLUE),
         ("交付", "模型对比接口、示例题、训练说明文档，供苏学长做演示界面",
          "Comparison API, example questions, training notes for the UI", GOLD)]
for i, (h, body, en, col) in enumerate(cards):
    x, y = 0.6 + (i % 2) * 6.15, 1.85 + (i // 2) * 2.6
    box(s, x, y, 5.95, 2.35, CARD, RULE)
    circle_num(s, x + 0.3, y + 0.3, i + 1, col)
    text(s, x + 1.0, y + 0.3, 4.7, 0.5, [(h, {"size": 20, "bold": True, "font": NUM if h.startswith("S2") else CN})], anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, y + 1.0, 5.35, 1.3, [(body, {"size": 15, "color": SOFT, "space": 4}), (en, {"size": 12, "color": TAUPE, "font": NUM})])
notes(s, "第一，S2-6datasets：六个数据集只用训练部分，测试题一道都没进训练，VQA-RAD 还去掉了问测试图片的题。"
         "第二，S2-CheXpertPlus：只用一万对胸片和报告训练。第三，完整评测，漏报用 CheXbert 再核一遍。第四，给苏学长准备了对比接口和文档。")

# ================================================================ 3 结果表
s = slide(PAPER)
title(s, "四个模型对比", "Four 35B models compared (higher is better unless ↓)")
cols = [("模型", "Model"), ("SLAKE 是非题", "Medical VQA"), ("敢说「不确定」/55", "PubMedQA maybe"),
        ("病理图 PathVQA", "Pathology"), ("医师考试 MedQA", "Med knowledge"), ("胸片描述漏报 ↓", "Missed in description"),
        ("区分能力", "Discrimination")]
keys = [None, "slake", "maybe", "path", "medqa", "omit", "disc"]
best = {k: (min if k == "omit" else max)(V[m][k] for _, _, m, _ in MODELS) for k in keys[1:]}
tbl = s.shapes.add_table(len(MODELS) + 1, len(cols), Inches(0.6), Inches(1.9), Inches(12.1), Inches(4.0)).table
widths = [2.6, 1.55, 1.75, 1.55, 1.55, 1.6, 1.5]
for j, w in enumerate(widths):
    tbl.columns[j].width = Inches(w)


def cell(c, lines, fill, align=PP_ALIGN.CENTER):
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.margin_left = c.margin_right = Inches(0.08)
    c.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = c.text_frame
    tf.word_wrap = True
    for i, (t, size, color, bold, font) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(size), color, bold, font
        set_ea(r)


for j, (cn, en) in enumerate(cols):
    cell(tbl.cell(0, j), [(cn, 14, WHITE, True, CN), (en, 11, PALE, False, NUM)], NAVY, PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
for i, (cn, en, k, col) in enumerate(MODELS, 1):
    fill = CARD if i % 2 else DEEP
    cell(tbl.cell(i, 0), [(cn, 15, col, True, NUM if cn.startswith("S2") else CN), (en, 11, TAUPE, False, NUM)], fill, PP_ALIGN.LEFT)
    for j, key in enumerate(keys[1:], 1):
        v = V[k][key]
        txt = f"{v:.1f}%" if key == "omit" else (f"{v}" if key == "maybe" else f"{v:.1f}")
        is_best = v == best[key]
        cell(tbl.cell(i, j), [(txt, 18, GREEN if is_best else INK, is_best, NUM)], fill)
for i in range(len(MODELS) + 1):
    tbl.rows[i].height = Inches(0.8)
text(s, 0.6, 6.15, 12.1, 0.8, [("绿色为最好。外部数据集对 S2-6datasets 属于练过同类题（没练过测试题）；每个模型只训练一次。",
                               {"size": 13, "color": SOFT, "space": 2}),
                              ("Green = best. External sets are in-domain for S2-6datasets; single training run per model.",
                               {"size": 11, "color": TAUPE, "font": NUM})])
notes(s, "绿色是每一列最好的。S2-6datasets 在医学问答、病理图、医师考试、描述漏报上都是最好或接近最好；"
         "S2-CheXpertPlus 只练胸片报告，所以短答题回到原版水平，但胸片描述的区分能力最高。")

# ================================================================ 4 主要发现
s = slide(PAPER)
title(s, "主要发现", "Key findings")
box(s, 0.6, 1.8, 6.2, 5.1, CARD, RULE)
cd = CategoryChartData()
cd.categories = ["原版", "S2-2datasets", "S2-6datasets", "S2-CheXpertPlus"]
cd.add_series("描述漏报 %", [V[k]["omit"] for _, _, k, _ in MODELS])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(2.45), Inches(5.8), Inches(4.3), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = False
plot = ch.plots[0]
plot.gap_width = 70
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = '0.0"%"', False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.name, dl.font.color.rgb = Pt(14), True, NUM, INK
for idx, (_, _, _, col) in enumerate(MODELS):
    pt = plot.series[0].points[idx]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
va = ch.value_axis
va.maximum_scale, va.minimum_scale = 80, 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RULE
va.tick_labels.font.size, va.tick_labels.font.color.rgb, va.tick_labels.font.name = Pt(11), TAUPE, NUM
va.format.line.fill.background()
ca = ch.category_axis
ca.tick_labels.font.size, ca.tick_labels.font.color.rgb, ca.tick_labels.font.name = Pt(12), SOFT, NUM
ca_rpr = ca.tick_labels.font._rPr  # 「原版」用雅黑, 英文名仍是 Arial, 标签不会被挤成斜的
ca_rpr.append(ca_rpr.makeelement(qn("a:ea"), {"typeface": CN}))
ca.format.line.color.rgb = RULE
text(s, 0.85, 1.95, 5.8, 0.5, [("成人胸片：病片被描述成「正常」的比例", {"size": 15, "bold": True, "space": 0}),
                               ("Abnormal X-rays described as normal (3,189 films) ↓", {"size": 11, "color": TAUPE, "font": NUM})])
finds = [("数据决定可靠性", "加入更多种类的医学数据，或专门的胸片报告，描述漏报都从 68.5% 降到 14% 左右",
          "Data, not architecture, fixes missed findings", GREEN),
         ("各有所长", "S2-6datasets 最全面；S2-CheXpertPlus 胸片描述最平衡、误报 " + f"{V['S2-CheXpertPlus']['falarm']:.0f}%" + "；其他问答回到原版水平",
          "6datasets: all-round · CheXpertPlus: best on X-ray reports only", PURPLE),
         ("判断方法可信", f"CheXbert 复核 S2-6datasets：漏报 {judge['abnormal_no_finding_reported_pct']:.1f}%，与关键词判断逐条一致 {judge['agreement_with_keyword_judge_pct']:.1f}%",
          "Findings hold with an independent judge", BLUE)]
for i, (h, body, en, col) in enumerate(finds):
    y = 1.8 + i * 1.75
    box(s, 7.1, y, 5.65, 1.55, CARD, RULE)
    circle_num(s, 7.35, y + 0.25, i + 1, col)
    text(s, 8.05, y + 0.2, 4.5, 1.3, [(h, {"size": 17, "bold": True, "space": 2}), (body, {"size": 14, "color": SOFT, "space": 2}),
                                      (en, {"size": 11, "color": TAUPE, "font": NUM})])
notes(s, "左边是最关键的一张图：病片被描述成正常的比例。现在部署的 S2-2datasets 是 68.5%，两个新模型都降到 14% 左右。"
         "结论是可靠性主要靠训练数据。两个新模型各有所长。另外 CheXbert 独立复核，结论成立。")

# ================================================================ 5 下一步
s = slide(NAVY)
title(s, "下一步", "Next steps", dark=True)
steps = [("上线 S2-6datasets", "替换现在部署的模型（等世龙同意，不用动他的文件）", "Deploy S2-6datasets in the shared service"),
         ("论文系统功能", "一致性检查：问答与描述矛盾时提醒；回答「不确定」时提示人工复核", "Reliability features: consistency check, uncertainty alert"),
         ("中文问答", "SLAKE 自带中文题（测试 1,033 道），先测中文表现，再加中文与翻译数据训练", "Chinese QA with SLAKE-zh and translated data"),
         ("结果确认", "用种子 43 复训 S2-6datasets；CheXbert 判完全部模型；论文题目 10/16 定稿", "Second seed, full CheXbert scoring, thesis title")]
for i, (h, body, en) in enumerate(steps):
    y = 1.85 + i * 1.28
    box(s, 0.6, y, 12.1, 1.1, NAVY2)
    circle_num(s, 0.9, y + 0.3, i + 1, [GREEN, PURPLE, BLUE, GOLD][i])
    text(s, 1.65, y + 0.15, 3.3, 0.8, [(h, {"size": 18, "bold": True, "color": WHITE})], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 5.0, y + 0.1, 7.5, 0.9, [(body, {"size": 15, "color": WHITE, "space": 2}), (en, {"size": 11, "color": PALE, "font": NUM})], anchor=MSO_ANCHOR.MIDDLE)
notes(s, "下一步四件事：一，经世龙同意后把线上模型换成 S2-6datasets；二，为论文做两个可靠性功能，一致性检查和不确定提示；"
         "三，中文问答，SLAKE 本来就有中文题，先测再训；四，用第二个种子确认结果，CheXbert 判完全部模型，论文题目 16 号定稿。")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("written", OUT)
