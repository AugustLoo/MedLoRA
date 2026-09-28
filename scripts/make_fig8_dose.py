"""图 8: 回放剂量曲线 (0 / 99 / 300 / 900 条) 在 3B 与 35B 上的对比 (PubMedQA held-out 500 题)。
数据来源: outputs/eval/pubmedqa_<tag>_preds.jsonl, 按 data/pubmedqa_split.json 的 test 半边过滤。
线 = 种子均值, 点 = 每次运行 (3B 的 0 与 99 各只有一次)。
用法: python scripts/make_fig8_dose.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
EV, OUT = REPO / "outputs" / "eval", REPO / "docs" / "figures"
LABELS = ("yes", "no", "maybe")

# 与 make_figures.py / make_fig7_scale.py 同一套配色和字体
COL = {"3B": "#2a78d6", "35B": "#eb6834"}
GOLD = "#9a9994"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})

BUDGETS = [0, 99, 300, 900]
RUNS = {
    "3B": {0: ["sft_a_server"], 99: ["sft_mix_100"], 300: ["sft_mix_300", "sft_mix_300_s43"],
           900: ["sft_mix_pubmedqa_r16", "sft_mix_900_s43"]},
    "35B": {0: ["interns2_mix_0", "interns2_abl_r0_s43"], 99: ["interns2_abl_r100", "interns2_abl_r100_s43"],
            300: ["interns2_mix_300", "interns2_abl_s43"], 900: ["interns2_abl_r900", "interns2_abl_r900_s43"]},
}
keep = set(json.loads((REPO / "data" / "pubmedqa_split.json").read_text(encoding="utf-8"))["test"])


def metrics(tag):
    rows = [json.loads(l) for l in open(EV / f"pubmedqa_{tag}_preds.jsonl", encoding="utf-8")]
    pairs = [(r["gold"], r["pred"]) for r in rows if int(r["pubid"]) in keep]
    assert len(pairs) == 500, (tag, len(pairs))
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"f1": 100 * sum(f1s) / 3, "acc": 100 * sum(g == p for g, p in pairs) / len(pairs),
            "pm": sum(p == "maybe" for _, p in pairs)}


data = {m: {b: [metrics(t) for t in tags] for b, tags in bs.items()} for m, bs in RUNS.items()}

PANELS = [("f1", "PubMedQA macro-F1"), ("pm", 'predicted "maybe" (gold: 55)'), ("acc", "PubMedQA accuracy")]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
xs = range(len(BUDGETS))
for ax, (key, title) in zip(axes, PANELS):
    if key == "pm":
        ax.axhline(55, color=GOLD, linestyle="--", linewidth=1, zorder=1)
        ax.text(len(BUDGETS) - 1.05, 57, "gold", color=GOLD, fontsize=7, ha="right", va="bottom")
    for m in ("3B", "35B"):
        means = []
        for i, b in enumerate(BUDGETS):
            vals = [r[key] for r in data[m][b]]
            means.append(sum(vals) / len(vals))
            ax.scatter([i] * len(vals), vals, s=16, facecolor="white", edgecolor=COL[m], linewidth=1.1, zorder=3)
        ax.plot(list(xs), means, color=COL[m], linewidth=2, marker="o", markersize=4, label=m, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("replay examples in the SFT mix", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, zorder=0)
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig8_dose.png")

for m in data:
    for b in BUDGETS:
        print(m, b, [(round(r["f1"], 2), round(r["acc"], 1), r["pm"]) for r in data[m][b]])
print("->", OUT / "fig8_dose.png")
