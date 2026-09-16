"""生成报告用图到 docs/figures/。数据来源: results/*.json, outputs/eval/*_preds.jsonl, outputs/<exp>/trainer_log.jsonl。
用法: python scripts/make_figures.py
有哪个实验的数据就画哪个 (B2 缺就自动跳过那一列)。
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES, EV, OUT = REPO / "results", REPO / "outputs" / "eval", REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 固定顺序的分类色 (dataviz 参考色板, 相邻对色盲安全), 文本一律用墨色
COL = {"Base": "#2a78d6", "A": "#eb6834", "B1": "#1baf7a", "B2": "#eda100"}
GOLD = "#9a9994"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})

MODELS = [("Base", "baseline_2026-09-15.json", "baseline"),
          ("A", "sft_A_2026-09-15.json", "sft_r16"),
          ("B1", "cpt_sft_B_2026-09-16.json", "cpt_sft_r16"),
          ("B2", "cpt_iu_sft_B2_2026-09-16.json", "cpt_iu_sft_r16")]
LABEL = {"Base": "Base (zero-shot)", "A": "A: SFT", "B1": "B1: text CPT → SFT", "B2": "B2: image CPT → SFT"}

res, tags = {}, {}
for name, fn, tag in MODELS:
    p = RES / fn
    if not p.exists():
        alt = [EV / f"{k}_{tag}.json" for k in ("slake", "textvqa", "pubmedqa")]
        if all(a.exists() for a in alt):
            res[name] = {k: json.load(open(a)) for k, a in zip(("slake", "textvqa", "pubmedqa"), alt)}
    else:
        res[name] = json.load(open(p))
    if name in res:
        tags[name] = tag
names = [n for n, _, _ in MODELS if n in res]
print("models:", names)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------- Fig 2: main results ----------
metrics = [("SLAKE closed acc", lambda r: r["slake"]["metrics"]["closed_acc"]),
           ("SLAKE open recall", lambda r: r["slake"]["metrics"]["open_recall"]),
           ("TextVQA acc", lambda r: r["textvqa"]["textvqa_acc"]),
           ("PubMedQA acc", lambda r: r["pubmedqa"]["accuracy"]),
           ("PubMedQA macro-F1", lambda r: r["pubmedqa"]["macro_f1"])]
fig, ax = plt.subplots(figsize=(8.4, 3.6))
w = 0.8 / len(names)
for i, n in enumerate(names):
    xs = [j + (i - (len(names) - 1) / 2) * w for j in range(len(metrics))]
    ys = [f(res[n]) for _, f in metrics]
    ax.bar(xs, ys, width=w * 0.92, color=COL[n], label=LABEL[n], zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1, f"{y:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
ax.set_xticks(range(len(metrics)), [m for m, _ in metrics])
ax.set_ylim(0, 105)
ax.set_ylabel("score (%)")
ax.legend(frameon=False, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8)
style(ax)
fig.savefig(OUT / "fig2_main_results.png")
plt.close(fig)

# ---------- Fig 3: SLAKE by question type ----------
def load_preds(tag):
    p = EV / f"slake_{tag}_preds.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p, encoding="utf-8")} if p.exists() else None

preds = {n: load_preds(tags[n]) for n in names}
preds = {n: p for n, p in preds.items() if p}
if "Base" in preds and "A" in preds:
    groups = defaultdict(list)
    for q, r in preds["Base"].items():
        groups[(r["content_type"], r["answer_type"])].append(q)
    rows = [(k, v) for k, v in groups.items() if len(v) >= 20]
    rows.sort(key=lambda kv: -len(kv[1]))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.2))
    for yi, (k, qs) in enumerate(rows):
        y = len(rows) - 1 - yi
        vals = {n: 100 * sum(preds[n][q]["score"] for q in qs) / len(qs) for n in preds}
        ax.plot([vals["Base"], vals["A"]], [y, y], color=GRID, lw=2, zorder=2)
        for n in preds:
            ax.scatter(vals[n], y, s=34, color=COL[n], zorder=3, edgecolor="white", linewidth=0.8)
        right = max(v for n, v in vals.items() if n != "Base")
        ax.text(right + 1.8, y, f"{vals['A']:.0f}", va="center", fontsize=7, color=INK2)
        ax.text(min(vals.values()) - 1.8, y, f"{vals['Base']:.0f}", va="center", ha="right", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{k[0]} ({k[1].lower()}), n={len(qs)}" for k, qs in reversed(rows)])
    ax.set_xlim(0, 105)
    ax.set_xlabel("accuracy (%), open questions scored by exact match")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[n], label=LABEL[n]) for n in preds]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(preds))
    ax.set_xlim(0, 110)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig3_slake_by_type.png")
    plt.close(fig)

# ---------- Fig 4: training curves ----------
runs = [("A: SFT", "sft_slake_qlora_r16", "A"), ("B1: text CPT", "cpt_pubmed_qlora_r16", "B1"),
        ("B1: SFT after CPT", "sft_after_cpt_r16", "B1"), ("B2: image CPT", "cpt_iu_qlora_r16", "B2"),
        ("B2: SFT after CPT", "sft_after_cpt_iu_r16", "B2")]
runs = [(t, d, n) for t, d, n in runs if (REPO / "outputs" / d / "trainer_log.jsonl").exists()]
if runs:
    fig, axes = plt.subplots(1, len(runs), figsize=(2.6 * len(runs) + 0.8, 2.8), sharey=False)
    axes = [axes] if len(runs) == 1 else list(axes)
    for ax, (title, d, n) in zip(axes, runs):
        rows = [json.loads(l) for l in open(REPO / "outputs" / d / "trainer_log.jsonl")]
        tr = [(r["current_steps"], r["loss"]) for r in rows if "loss" in r]
        ev = [(r["current_steps"], r["eval_loss"]) for r in rows if "eval_loss" in r]
        ax.plot(*zip(*tr), color=COL[n], lw=1.6, label="train")
        if ev:
            ax.plot(*zip(*ev), color=INK, lw=0, marker="o", ms=4, label="validation")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("step"); style(ax)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("loss")
    fig.savefig(OUT / "fig4_training_curves.png")
    plt.close(fig)

# ---------- Fig 5: PubMedQA predicted label distribution ----------
labels = ["yes", "no", "maybe"]
gold = res[names[0]]["pubmedqa"]["gold_dist"]
series = [("Gold", gold, GOLD)] + [(n, res[n]["pubmedqa"]["pred_dist"], COL[n]) for n in names]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
w = 0.8 / len(series)
for i, (n, dist, c) in enumerate(series):
    xs = [j + (i - (len(series) - 1) / 2) * w for j in range(3)]
    ys = [dist[l] for l in labels]
    ax.bar(xs, ys, width=w * 0.92, color=c, label=LABEL.get(n, "Gold labels"), zorder=3,
           hatch="////" if n == "Gold" else None, edgecolor="white" if n == "Gold" else None, linewidth=0.4)
    for x, y in zip(xs, ys):
        ax.text(x, y + 8, str(y), ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_xticks(range(3), labels)
ax.set_ylabel("number of questions (of 1,000)")
ax.legend(frameon=False, fontsize=8, ncol=2)
style(ax)
fig.savefig(OUT / "fig5_pubmedqa_distribution.png")
plt.close(fig)

print("written:", sorted(p.name for p in OUT.glob("*.png")))
