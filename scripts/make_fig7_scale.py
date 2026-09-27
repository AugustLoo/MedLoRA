"""图 7: 3B 与 35B 在「基座 / 回放 0 / 回放 300」三个条件下的 PubMedQA 行为对比 (held-out 500 题)。
数据来源: outputs/eval/pubmedqa_<tag>_preds.jsonl, 按 data/pubmedqa_split.json 的 test 半边过滤。
柱 = 种子均值, 点 = 每个种子 (3B 回放 0 只有一次运行)。
用法: python scripts/make_fig7_scale.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
EV, OUT = REPO / "outputs" / "eval", REPO / "docs" / "figures"
LABELS = ("yes", "no", "maybe")

# 与 make_figures.py 同一套配色和字体
COL = {"3B": "#2a78d6", "35B": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})

CONDS = ["Base", "Replay 0", "Replay 300"]
RUNS = {
    "3B": {"Base": ["baseline_server"], "Replay 0": ["sft_a_server"],
           "Replay 300": ["sft_mix_300", "sft_mix_300_s43"]},
    "35B": {"Base": ["interns2_base"], "Replay 0": ["interns2_mix_0", "interns2_abl_r0_s43"],
            "Replay 300": ["interns2_mix_300", "interns2_abl_s43"]},
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
    return {"f1": 100 * sum(f1s) / 3,
            "maybe": sum(g == "maybe" and p == "maybe" for g, p in pairs),
            "n2y": sum(g == "no" and p == "yes" for g, p in pairs),
            "y2n": sum(g == "yes" and p == "no" for g, p in pairs)}


data = {m: {c: [metrics(t) for t in tags] for c, tags in conds.items()} for m, conds in RUNS.items()}

PANELS = [("f1", "PubMedQA macro-F1", "higher is better"),
          ("maybe", '"maybe" answered correctly (of 55)', "higher is better"),
          ("n2y", 'gold "no" answered "yes"', "lower is better"),
          ("y2n", 'gold "yes" answered "no"', "lower is better")]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
w = 0.36
for ax, (key, title, hint) in zip(axes, PANELS):
    for i, m in enumerate(("3B", "35B")):
        for j, c in enumerate(CONDS):
            vals = [r[key] for r in data[m][c]]
            x = j + (i - 0.5) * w
            mean = sum(vals) / len(vals)
            ax.bar(x, mean, w * 0.92, color=COL[m], alpha=0.85, label=m if j == 0 else None, zorder=2)
            ax.scatter([x] * len(vals), vals, s=14, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
            top = max(vals + [mean])
            ax.text(x, top + (0.9 if key == "f1" else 1.3), f"{mean:.1f}" if key == "f1" else f"{mean:g}",
                    ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(CONDS)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(hint, fontsize=7, color=INK2)
    ax.grid(axis="y", color=GRID, zorder=0)
    if key == "f1":
        ax.set_ylim(40, 72)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig7_scale.png")

for m in data:
    for c in CONDS:
        print(m, c, [(round(r["f1"], 2), r["maybe"], r["n2y"], r["y2n"]) for r in data[m][c]])
print("->", OUT / "fig7_scale.png")
