"""给做对比界面的同学挑 S2-6datasets 的示例题: 从已有评测记录里选「S2-6datasets 答对、原版和 S2-2datasets 都答错」的题。

图文 5 条 (SLAKE 测试 3 条 + PneumoniaMNIST 测试 2 张) + 纯文字 5 条 (PubMedQA 考卷半边 2 条 + MedQA 测试 3 条)。
提问文字与评测时逐字相同, 在对比服务上重问应得到同样的答案 (贪心解码)。三个模型的记录答案都列出来。
SLAKE 图片写服务器路径; PneumoniaMNIST 图片从测试集 npz 导出成 PNG (MedMNIST, CC BY 4.0), 放在本目录 images/ 下, 一并部署到服务器。

输出: docs/compare_examples_6ds/{examples.jsonl, README.md, images/}
用法 (本机): python scripts/make_compare_examples_6ds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
from medvlm.prompts import MEDQA, PUBMEDQA, slake_prompt  # noqa: E402
from eval_pneumonia import Q_CLOSED, Q_FREE, reports_lung_abnormality  # noqa: E402

EV = REPO / "outputs" / "eval"
OUT = REPO / "docs" / "compare_examples_6ds"
SERVER_SLAKE = "/home/ubuntu/chunqian/data/SLAKE/imgs"
SERVER_HERE = "/home/ubuntu/chunqian/MedLoRA/docs/compare_examples_6ds/images"
MEDQA_ARROW = ("C:/Users/Administrator/.cache/huggingface/datasets/json/default-2f13a700dda856fc/0.0.0/"
               "915409a0dcfcf3c0e0e2b1d1e761f7eaf823c9e638ed16d3c75e0a940a971aa3/json-test.arrow")
TAGS = ("interns2_base", "interns2_mix_300", "interns2_6ds")
NAMES = ("原版", "S2-2datasets", "S2-6datasets")

SLAKE_PICK = [(12050, "胸片诊断：另外两个都说肺不张，S2-6datasets 答对肺炎"),
              (12654, "MRI 定位：原版说右叶、S2-2datasets 说右上叶，S2-6datasets 答对左上叶"),
              (12925, "CT 常识判断：另外两个都说有能运送食物的器官，S2-6datasets 答对「没有」")]
PNEU_PICK = [(92, "closed", "儿童胸片判断肺炎：另外两个都说没有，S2-6datasets 答对「有」"),
             (105, "free", "儿童胸片描述：另外两个都写肺野清晰，S2-6datasets 说出右肺有实变")]
PUBMEDQA_N, MEDQA_N = 2, 3


def load(prefix, key):
    return [{r[key]: r for r in (json.loads(x) for x in open(EV / f"{prefix}_{t}_preds.jsonl", encoding="utf-8"))}
            for t in TAGS]


def main():
    rows = []
    meta = {r["qid"]: r for r in json.loads((REPO / "data/raw/SLAKE/test.json").read_text(encoding="utf-8"))}
    sl = load("slake", "qid")
    for n, (qid, why) in enumerate(SLAKE_PICK, 1):
        m = meta[qid]
        assert [p[qid]["score"] for p in sl] == [0, 0, 1], qid
        rows.append({"id": f"img{n}", "kind": "图文", "source": f"SLAKE 测试集 qid {qid} ({m['modality']})",
                     "question": slake_prompt(m["question"], m["answer_type"]),
                     "image": f"{SERVER_SLAKE}/{m['img_name']}", "gold": m["answer"],
                     **{nm: p[qid]["pred"] for nm, p in zip(NAMES, sl)}, "why": why})

    pn = load("pneumonia", "i")
    x = np.load(REPO / "data/raw/medmnist/pneumoniamnist_224_test.npz")["test_images"]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for n, (i, kind, why) in enumerate(PNEU_PICK, len(SLAKE_PICK) + 1):
        assert pn[2][i]["label"] == "pneumonia" and [p[i]["closed"] for p in pn] == ["no", "no", "yes"], i
        assert [reports_lung_abnormality(p[i]["desc"]) for p in pn] == [False, False, True], i
        fn = f"pneumoniamnist_test_{i}.png"
        Image.fromarray(x[i]).save(OUT / "images" / fn)
        q, ans = (Q_CLOSED, "closed_raw") if kind == "closed" else (Q_FREE, "desc")
        rows.append({"id": f"img{n}", "kind": "图文", "source": f"PneumoniaMNIST 测试集第 {i} 张 (儿童胸片, 标签: 肺炎)",
                     "question": q, "image": f"{SERVER_HERE}/{fn}", "local_image": f"images/{fn}",
                     "gold": "yes" if kind == "closed" else "（应指出肺部有异常 / 肺炎）",
                     **{nm: p[i][ans].strip() for nm, p in zip(NAMES, pn)}, "why": why})

    from datasets import Dataset, load_dataset
    pq = {int(r["pubid"]): r for r in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")}
    held = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
    pp = load("pubmedqa", "pubid")
    cand = [k for k in pp[2] if k in held and pp[2][k]["pred"] == pp[2][k]["gold"] == "maybe"
            and pp[0][k]["pred"] != "maybe" and pp[1][k]["pred"] != "maybe"]
    cand.sort(key=lambda k: len(" ".join(pq[k]["context"]["contexts"])))
    for n, k in enumerate(cand[:PUBMEDQA_N], 1):
        r = pq[k]
        rows.append({"id": f"txt{n}", "kind": "纯文字", "source": f"PubMedQA 考卷半边 pubid {k}",
                     "question": PUBMEDQA.format(context="\n".join(r["context"]["contexts"]), question=r["question"]),
                     "short_question": r["question"], "gold": "maybe",
                     **{nm: p[k]["pred"] for nm, p in zip(NAMES, pp)},
                     "why": "证据不足时敢说「不确定」：另外两个都给了肯定答案"})

    mq = Dataset.from_file(MEDQA_ARROW)
    mp = load("medqa", "i")
    mc = [i for i in mp[2] if [p[i]["score"] for p in mp] == [0, 0, 1]]
    mc.sort(key=lambda i: len(mq[i]["question"]))
    for n, i in enumerate(mc[:MEDQA_N], PUBMEDQA_N + 1):
        r = mq[i]
        assert r["answer_idx"] == mp[2][i]["gold"], i
        opts = {c: str(r["options"][c]).strip() for c in "ABCD"}
        rows.append({"id": f"txt{n}", "kind": "纯文字", "source": f"MedQA 测试集第 {i} 题",
                     "question": MEDQA.format(question=str(r["question"]).strip(),
                                              options="\n".join(f"{c}. {v}" for c, v in opts.items())),
                     "gold": f"{r['answer_idx']}. {opts[r['answer_idx']]}",
                     **{nm: p[i]["pred"] for nm, p in zip(NAMES, mp)},
                     "why": f"医师考试病例题：原版选 {mp[0][i]['pred']}、S2-2datasets 选 {mp[1][i]['pred']}，S2-6datasets 选对"})

    with open(OUT / "examples.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_readme(rows)
    print(f"{len(rows)} 条 → {OUT}")


def write_readme(rows):
    md = ["# S2-6datasets 示例题：三个模型对比", "",
          "给做对比界面的同学。10 道题都来自公开测试集，**S2-6datasets 答对、原版和 S2-2datasets 都答错**。",
          "", "- 原版 = `intern-s2-base`；S2-2datasets = `intern-s2-medlora`（世龙部署的那个）；S2-6datasets = `intern-s2-6ds`。",
          "- 下面的回答是评测时记录的（关闭思考模式、temperature 0），用对比接口按同样的提问重问，答案应该一样或几乎一样。",
          "- **提问请用「完整提问」的原文**，包括末尾那句回答要求。",
          "- 这些题是**挑出来的**；整体成绩：S2-6datasets 在医学看图问答、「不确定」题、医学通用能力上与 S2-2datasets 持平，"
          "在病理图（PathVQA）、医师考试题（MedQA）、儿童胸片判断肺炎、描述胸片时不漏报这几项上明显更好。",
          "- 批量重跑：`python scripts/compare_ask.py batch docs/compare_examples_6ds/examples.jsonl`", "",
          "## 图文题（5 道）", ""]
    for r in rows:
        if r["kind"] != "图文":
            continue
        md += [f"### {r['id']} · {r['why']}", "", f"- 来源：{r['source']}", f"- 图片（服务器）：`{r['image']}`"]
        if r.get("local_image"):
            md += ["", f"![{r['id']}]({r['local_image']})"]
        md += ["- 完整提问：", "", "```", r["question"], "```", "", f"- **正确答案**：{r['gold']}", ""]
        long = any(len(str(r[n])) > 40 for n in NAMES)
        if long:
            for n, mark in zip(NAMES, ("❌", "❌", "✅")):
                md += [f"**{n}** {mark}", "", f"> {r[n]}", ""]
        else:
            md += ["| 原版 | S2-2datasets | S2-6datasets |", "|---|---|---|",
                   f"| {r['原版']} ❌ | {r['S2-2datasets']} ❌ | {r['S2-6datasets']} ✅ |", ""]
    md += ["## 纯文字题（5 道）", "", "PubMedQA 题先给一段论文摘要（Context），再问问题，要整段一起复制去问。", ""]
    for r in rows:
        if r["kind"] != "纯文字":
            continue
        md += [f"### {r['id']} · {r['why']}", "", f"- 来源：{r['source']}"]
        if "short_question" in r:
            md += [f"- 问题：*{r['short_question']}*"]
        md += ["- 完整提问：", "", "```", r["question"], "```", "", f"- **正确答案**：{r['gold']}", "",
               "| 原版 | S2-2datasets | S2-6datasets |", "|---|---|---|",
               f"| {r['原版']} ❌ | {r['S2-2datasets']} ❌ | {r['S2-6datasets']} ✅ |", ""]
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
