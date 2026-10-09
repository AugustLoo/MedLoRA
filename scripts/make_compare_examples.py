"""给做对比界面的同学挑示例题: 从已有评测记录里选「训练版答对、原版答错」的题, 图文 5 条 + 纯文字 5 条。

题目原样来自公开测试集 (SLAKE 测试、PubMedQA 考卷半边、MedQA 测试), 提问文字与评测时逐字相同,
所以在对比服务 (train/interns2/compare_serve.sh) 上重问, 答案应与这里记录的一致 (贪心解码)。
图片路径写服务器上的 (/home/ubuntu/chunqian/data/SLAKE/imgs/...), 本机只有 3 张 SLAKE 图。

输出: docs/compare_examples/examples.jsonl (scripts/compare_ask.py batch 可直接读) 与 README.md
用法 (本机): python scripts/make_compare_examples.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.prompts import MEDQA, PUBMEDQA, slake_prompt  # noqa: E402

EV = REPO / "outputs" / "eval"
OUT = REPO / "docs" / "compare_examples"
SERVER_SLAKE = "/home/ubuntu/chunqian/data/SLAKE/imgs"
MEDQA_ARROW = ("C:/Users/Administrator/.cache/huggingface/datasets/json/default-2f13a700dda856fc/0.0.0/"
               "915409a0dcfcf3c0e0e2b1d1e761f7eaf823c9e638ed16d3c75e0a940a971aa3/json-test.arrow")

SLAKE_PICK = [  # (qid, 为什么值得演示)
    (12070, "看胸片诊断：原版说成肺不张，训练版答对气胸"),
    (12218, "认器官：图像左边是病人的右侧，原版答成肝，训练版答对右肺"),
    (12620, "数异常：原版数成 1 种，训练版答对 2 种"),
    (12029, "判断健康：原版说肺是健康的，训练版答对「不健康」"),
    (12582, "判断有没有某个器官：原版说有食管，训练版答对「没有」"),
]
PUBMEDQA_PICK = [
    (17076091, "证据不足时敢说「不确定」：原版答 yes"),
    (12920330, "证据不足时敢说「不确定」：原版答 yes"),
    (22954812, "证据不足时敢说「不确定」：原版答 yes"),
]
MEDQA_PICK = [
    (300, "医师考试病例题：原版选 B，训练版选对 D"),
    (477, "医师考试病例题：原版选 D，训练版选对 A"),
]


def load_jsonl(path, key):
    return {r[key]: r for r in (json.loads(x) for x in open(path, encoding="utf-8"))}


def main():
    rows = []
    meta = {r["qid"]: r for r in json.loads((REPO / "data/raw/SLAKE/test.json").read_text(encoding="utf-8"))}
    sb = load_jsonl(EV / "slake_interns2_base_preds.jsonl", "qid")
    sf = load_jsonl(EV / "slake_interns2_mix_300_preds.jsonl", "qid")
    for n, (qid, why) in enumerate(SLAKE_PICK, 1):
        m = meta[qid]
        assert sf[qid]["score"] == 1 and sb[qid]["score"] == 0, qid
        rows.append({"id": f"img{n}", "kind": "图文", "source": f"SLAKE 测试集 qid {qid} ({m['modality']})",
                     "question": slake_prompt(m["question"], m["answer_type"]),
                     "image": f"{SERVER_SLAKE}/{m['img_name']}", "gold": m["answer"],
                     "intern-s2-base": sb[qid]["pred"], "intern-s2-medlora": sf[qid]["pred"], "why": why})

    from datasets import Dataset, load_dataset
    pq = {int(r["pubid"]): r for r in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")}
    pb = load_jsonl(EV / "pubmedqa_interns2_base_preds.jsonl", "pubid")
    pf = load_jsonl(EV / "pubmedqa_interns2_mix_300_preds.jsonl", "pubid")
    held_out = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
    for n, (pid, why) in enumerate(PUBMEDQA_PICK, 1):
        r = pq[pid]
        assert pid in held_out and pf[pid]["pred"] == r["final_decision"] != pb[pid]["pred"], pid
        prompt = PUBMEDQA.format(context="\n".join(r["context"]["contexts"]), question=r["question"])
        rows.append({"id": f"txt{n}", "kind": "纯文字", "source": f"PubMedQA 考卷半边 pubid {pid}",
                     "question": prompt, "short_question": r["question"], "gold": r["final_decision"],
                     "intern-s2-base": pb[pid]["pred"], "intern-s2-medlora": pf[pid]["pred"], "why": why})

    mq = Dataset.from_file(MEDQA_ARROW)
    mb = load_jsonl(EV / "medqa_interns2_base_preds.jsonl", "i")
    mf = load_jsonl(EV / "medqa_interns2_mix_300_preds.jsonl", "i")
    for n, (i, why) in enumerate(MEDQA_PICK, len(PUBMEDQA_PICK) + 1):
        r = mq[i]
        assert mf[i]["score"] == 1 and mb[i]["score"] == 0 and r["answer_idx"] == mf[i]["gold"], i
        opts = {k: str(r["options"][k]).strip() for k in "ABCD"}
        prompt = MEDQA.format(question=str(r["question"]).strip(), options="\n".join(f"{k}. {v}" for k, v in opts.items()))
        rows.append({"id": f"txt{n}", "kind": "纯文字", "source": f"MedQA 测试集第 {i} 题",
                     "question": prompt, "gold": f"{r['answer_idx']}. {opts[r['answer_idx']]}",
                     "intern-s2-base": mb[i]["pred"], "intern-s2-medlora": mf[i]["pred"], "why": why})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "examples.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_readme(rows)
    print(f"{len(rows)} 条 → {OUT}")


def write_readme(rows):
    md = ["# 对比示例题：原版 vs 训练版", "",
          "给做对比界面的同学。10 道题都来自公开测试集，**训练版答对、原版答错**，适合演示两个模型的差别。",
          "",
          "- 「原版」= `intern-s2-base`，「训练版」= `intern-s2-medlora`（交给世龙部署的那个）。",
          "- 下面的回答是我们评测时记录的（关闭思考模式、temperature 0）。用对比接口按同样的提问重问，答案应该一样或几乎一样。",
          "- **提问要用「完整提问」那一栏的原文**，包括末尾那句回答要求（例如 *Answer with yes or no only.*）。改了措辞，答案可能不同。",
          "- 这些题是**挑出来的**。整体成绩见 `docs/COMPARE_API.md` 第二节：训练版在医学看图问答和「不确定」题上明显更好，"
          "在 MedQA 上整体反而略低（85.0 → 83.6）。",
          "- 批量重跑：`python scripts/compare_ask.py batch docs/compare_examples/examples.jsonl`（文件里多出来的字段会被忽略）。",
          "", "## 图文题（5 道，SLAKE 测试集）", ""]
    for r in rows:
        if r["kind"] != "图文":
            continue
        md += [f"### {r['id']} · {r['why']}", "",
               f"- 来源：{r['source']}",
               f"- 图片（服务器）：`{r['image']}`",
               "- 完整提问：", "", "```", r["question"], "```", "",
               "| 正确答案 | 原版 | 训练版 |", "|---|---|---|",
               f"| {r['gold']} | {r['intern-s2-base']} ❌ | {r['intern-s2-medlora']} ✅ |", ""]
    md += ["## 纯文字题（5 道）", "",
           "PubMedQA 题先给一段论文摘要（Context），再问问题，要把下面整段一起复制去问。", ""]
    for r in rows:
        if r["kind"] != "纯文字":
            continue
        md += [f"### {r['id']} · {r['why']}", "", f"- 来源：{r['source']}"]
        if "short_question" in r:
            md += [f"- 问题：*{r['short_question']}*"]
        md += ["- 完整提问：", "", "```", r["question"], "```", ""]
        md += ["| 正确答案 | 原版 | 训练版 |", "|---|---|---|",
               f"| {r['gold']} | {r['intern-s2-base']} ❌ | {r['intern-s2-medlora']} ✅ |", ""]
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
