"""为 PPT 渲染与 docs/report.html 同配色的图表 PNG 到 docs/ppt/charts/。
用法: python scripts/make_ppt_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "ppt" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

CARD, PAPER = "#FDFAF2", "#F7F2E7"
NAVY, SOFT, TAUPE, RULE, IDLE = "#14263B", "#3D4E63", "#8C8577", "#D8CFBA", "#E4DBC4"
C = {"base": "#0E67B5", "a": "#C25E28", "b1": "#1F8A70", "b2": "#B8860B", "gold": "#9A9283"}
LABEL = {"base": "Base (zero-shot)", "a": "A: SFT", "b1": "B1: text CPT → SFT", "b2": "B2: image CPT → SFT"}
plt.rcParams.update({"font.family": ["Consolas", "DejaVu Sans Mono"], "font.size": 10, "axes.edgecolor": SOFT,
                     "axes.labelcolor": NAVY, "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD})

MAIN = [("SLAKE closed acc", {"base": 67.31, "a": 89.18, "b1": 88.70, "b2": 88.70}),
        ("SLAKE open recall", {"base": 46.73, "a": 82.17, "b1": 82.72, "b2": 81.61}),
        ("TextVQA acc", {"base": 83.89, "a": 83.89, "b1": 83.78, "b2": 84.00}),
        ("PubMedQA acc", {"base": 65.80, "a": 70.60, "b1": 72.80, "b2": 70.80}),
        ("PubMedQA macro-F1", {"base": 51.03, "a": 52.38, "b1": 54.19, "b2": 52.96})]
TYPES = [("Position (open)", 163, 24.5, 57.7, 60.1, 58.3), ("Organ (closed)", 154, 75.3, 90.9, 90.3, 89.6),
         ("Abnormality (closed)", 109, 68.8, 82.6, 82.6, 84.4), ("KG (open)", 109, 22.0, 72.5, 69.7, 70.6),
         ("Organ (open)", 99, 23.2, 84.8, 83.8, 84.8), ("Modality (open)", 75, 92.0, 94.7, 94.7, 94.7),
         ("Quantity (open)", 52, 71.2, 73.1, 76.9, 75.0), ("Abnormality (open)", 41, 12.2, 41.5, 39.0, 31.7),
         ("Size (open)", 39, 87.2, 97.4, 100, 100), ("KG (closed)", 39, 66.7, 69.2, 66.7, 66.7),
         ("Modality (closed)", 33, 57.6, 100, 100, 97.0), ("Plane (open)", 30, 53.3, 100, 100, 100),
         ("Color (open)", 30, 40.0, 100, 100, 100), ("Plane (closed)", 28, 78.6, 100, 100, 100),
         ("Size (closed)", 26, 26.9, 100, 100, 100), ("Position (closed)", 23, 65.2, 100, 100, 100)]
PUB = {"gold": {"yes": 552, "no": 338, "maybe": 110}, "base": {"yes": 641, "no": 201, "maybe": 158},
       "a": {"yes": 685, "no": 256, "maybe": 59}, "b1": {"yes": 670, "no": 286, "maybe": 44}, "b2": {"yes": 672, "no": 263, "maybe": 65}}


def style(ax, axis="y"):
    ax.grid(axis=axis, color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main_chart(models, fn, hi=None):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    w = 0.8 / 4
    for i, m in enumerate(["base", "a", "b1", "b2"]):
        shown = m in models
        xs = [j + (i - 1.5) * w for j in range(len(MAIN))]
        ys = [v[m] for _, v in MAIN]
        for j, (x, y) in enumerate(zip(xs, ys)):
            dim = hi is not None and hi != j
            ax.bar(x, y if shown else 0, width=w * 0.9, color=C[m] if shown else IDLE, alpha=0.3 if dim else 1, zorder=3,
                   label=LABEL[m] if j == 0 and shown else None)
            if shown:
                ax.text(x, y + 1.2, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=TAUPE if dim else NAVY, rotation=90)
    ax.set_xticks(range(len(MAIN)), [m for m, _ in MAIN], fontsize=10.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("score (%)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9.5)
    style(ax)
    fig.savefig(OUT / fn)
    plt.close(fig)


main_chart(["base"], "main_base.png")
main_chart(["base", "a"], "main_a.png")
main_chart(["base", "a", "b1"], "main_b1.png", hi=3)
main_chart(["base", "a", "b1", "b2"], "main_all.png")

# by type
fig, ax = plt.subplots(figsize=(9.6, 6.2))
n = len(TYPES)
for i, (t, cnt, b, a, b1, b2) in enumerate(TYPES):
    y = n - 1 - i
    ax.plot([b, a], [y, y], color=IDLE, lw=4, zorder=2)
    for v, m in [(b, "base"), (a, "a"), (b1, "b1"), (b2, "b2")]:
        ax.scatter(v, y, s=52, color=C[m], zorder=3, edgecolor=CARD, linewidth=1.2)
    ax.text(min(b, a, b1, b2) - 1.8, y, f"{b:.0f}", va="center", ha="right", fontsize=8.5, color=TAUPE)
    if t == "Abnormality (open)":
        ax.text(max(a, b1, b2) + 1.8, y, f"A {a:.1f} → B2 {b2:.1f}", va="center", fontsize=8.5, color="#C0392B", weight="bold")
    else:
        ax.text(max(a, b1, b2) + 1.8, y, f"{a:.0f}", va="center", fontsize=8.5, color=NAVY, weight="bold")
ax.set_yticks(range(n), [f"{t}  n={c}" for t, c, *_ in reversed(TYPES)], fontsize=9)
ax.set_xlim(0, 118)
ax.set_xlabel("accuracy (%), open questions scored by exact match")
handles = [plt.Line2D([], [], marker="o", ls="", color=C[m], label=LABEL[m]) for m in ["base", "a", "b1", "b2"]]
ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4)
style(ax, "x")
fig.savefig(OUT / "bytype.png")
plt.close(fig)

# pubmedqa
fig, ax = plt.subplots(figsize=(9.6, 4.2))
labels = ["yes", "no", "maybe"]
series = [("gold", "Gold labels"), ("base", LABEL["base"]), ("a", LABEL["a"]), ("b1", LABEL["b1"]), ("b2", LABEL["b2"])]
w = 0.8 / len(series)
for i, (k, lab) in enumerate(series):
    xs = [j + (i - 2) * w for j in range(3)]
    ys = [PUB[k][l] for l in labels]
    ax.bar(xs, ys, width=w * 0.9, color=C[k], alpha=0.55 if k == "gold" else 1, label=lab, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 8, str(y), ha="center", va="bottom", fontsize=8, color=NAVY)
ax.set_xticks(range(3), labels, fontsize=12)
ax.set_ylabel("number of questions (of 1,000)")
ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
style(ax)
fig.savefig(OUT / "pubmedqa.png")
plt.close(fig)

# training curves
runs = [("A · SFT", "sft_slake_qlora_r16", "a"), ("B1 · text CPT", "cpt_pubmed_qlora_r16", "b1"),
        ("B1 · SFT after CPT", "sft_after_cpt_r16", "b1"), ("B2 · image CPT", "cpt_iu_qlora_r16", "b2"),
        ("B2 · SFT after CPT", "sft_after_cpt_iu_r16", "b2")]
fig, axes = plt.subplots(1, 5, figsize=(13.2, 2.9))
for ax, (title, d, m) in zip(axes, runs):
    rows = [json.loads(l) for l in open(REPO / "outputs" / d / "trainer_log.jsonl")]
    tr = [(r["current_steps"], r["loss"]) for r in rows if "loss" in r]
    ev = [(r["current_steps"], r["eval_loss"]) for r in rows if "eval_loss" in r]
    ax.plot(*zip(*tr), color=C[m], lw=1.7, label="train")
    if ev:
        ax.plot(*zip(*ev), color=NAVY, lw=0, marker="o", ms=4.5, label="validation")
    ax.set_title(title, fontsize=10, color=NAVY, loc="left")
    ax.set_xlabel("step", fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("loss")
fig.tight_layout()
fig.savefig(OUT / "curves.png")
plt.close(fig)
print("charts:", sorted(p.name for p in OUT.glob("*.png")))
