"""下载 PubMedQA (HF: qiaojin/PubMedQA) 到本地缓存并打印概况。

pqa_labeled    : 1,000 条人工标注, 只用于评估 (幻觉/可靠性)
pqa_artificial : 211k 条自动标注, 用于 CPT 语料和文本 SFT
"""
from datasets import load_dataset


def main():
    for cfg in ["pqa_labeled", "pqa_artificial"]:
        ds = load_dataset("qiaojin/PubMedQA", cfg, split="train")
        print(cfg, len(ds), ds.column_names)
        r = ds[0]
        print("  示例:", r["question"], "->", r["final_decision"])


if __name__ == "__main__":
    main()
