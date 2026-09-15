"""把 PubMedQA pqa_artificial 转成 CPT 纯文本语料 (LLaMA-Factory stage=pt 格式)。

输出: data/processed/pubmed_cpt.json  [{"text": ...}, ...]
注意: 只用 pqa_artificial, 绝不把 pqa_labeled 混进训练, 那是评估集。
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20000, help="取多少条 (默认 2 万, 小规模 CPT)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    texts = []
    for i in idx[: args.max]:
        r = ds[i]
        ctx = "\n".join(r["context"]["contexts"])
        texts.append({"text": "Question: " + r["question"] + "\n\n" + ctx
                      + "\n\nConclusion: " + r["long_answer"]})

    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "pubmed_cpt.json"
    fn.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

    info_fn = OUT / "dataset_info.json"
    info = json.loads(info_fn.read_text(encoding="utf-8")) if info_fn.exists() else {}
    info["pubmed_cpt"] = {"file_name": fn.name, "columns": {"prompt": "text"}}
    info_fn.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(len(texts), "->", fn)


if __name__ == "__main__":
    main()
