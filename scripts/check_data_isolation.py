"""核对本机能核对的数据隔离: SLAKE 训练 / 测试的图片重叠, PubMedQA 两半的重叠,
并把 SLAKE 中文是非题按「图片是否在英文训练集里出现过」拆开重算。

用法: python scripts/check_data_isolation.py [--out results/xxx.json]
CheXpert Plus 的病人级隔离只能在服务器上核对 (病人号不出服务器), 不在这里。
"""
import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(name):
    return json.load(open(REPO / "data/raw/SLAKE" / name, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    tr, te = load("train.json"), load("test.json")
    imgs = {(s, lang): {r["img_id"] for r in rows if r["q_lang"] == lang}
            for s, rows in (("train", tr), ("test", te)) for lang in ("en", "zh")}
    seen = imgs[("train", "en")]                      # 训练只用英文部分
    zh_test = [r for r in te if r["q_lang"] == "zh"]
    out = {
        "slake_images": {f"{s}_{lang}": len(v) for (s, lang), v in imgs.items()},
        "en_train_vs_en_test_image_overlap": len(seen & imgs[("test", "en")]),
        "en_train_vs_zh_test_image_overlap": len(seen & imgs[("test", "zh")]),
        "zh_test_questions": len(zh_test),
        "zh_test_questions_on_en_train_images": sum(r["img_id"] in seen for r in zh_test),
    }
    split = json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))
    out["pubmedqa_train_test_overlap"] = len(set(split["train"]) & set(split["test"]))
    img_of = {r["qid"]: r["img_id"] for r in te}
    out["zh_closed_acc"] = {}
    for f in sorted(glob.glob(str(REPO / "outputs/eval/slake_zh_*_preds.jsonl"))):
        closed = [r for r in map(json.loads, open(f, encoding="utf-8")) if r["answer_type"] == "CLOSED"]
        groups = {"all": closed, "seen_images": [r for r in closed if img_of[r["qid"]] in seen],
                  "unseen_images": [r for r in closed if img_of[r["qid"]] not in seen]}
        out["zh_closed_acc"][Path(f).name[9:-12]] = {
            k: {"n": len(v), "acc": round(100 * sum(r["score"] for r in v) / len(v), 2)} for k, v in groups.items()}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
