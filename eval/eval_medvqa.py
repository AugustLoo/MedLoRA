"""外部医学 VQA 测试集: VQA-RAD 与 PathVQA (2026-09-30 老师同意加入, 只测不训)。

为什么: SLAKE 是训练用的同一个数据集, 分数高可能只是把 SLAKE 学熟了。
  - VQA-RAD: 放射科影像问答, 格式与 SLAKE 几乎一样, 但来源不同 → 检验换一个数据集还行不行。
  - PathVQA: 病理切片问答, 约一半是 yes/no 题 → 换一种影像, 并检验「爱答 yes」在看图题上是否也存在。

口径与 eval_slake.py 一致:
  - 标准答案是 yes / no 的算封闭题 (VQA-RAD / PathVQA 的 HF 版本没有 answer_type, 按 LLaVA-Med 的惯例以答案判断),
    用训练时同一句提示词 "Answer with yes or no only.", 按 yes/no 比对;
  - 其余算开放题, 提示词 "Answer with a single word or short phrase.", 报 exact match / token recall / token F1。
封闭题另外统计偏向: 预测 yes 的比例 vs 标准答案 yes 的比例, 「该 no 答 yes」「该 yes 答 no」各多少题, 答不出 yes/no 的多少题。

数据: HF flaviagiammarino/vqa-rad (test 451) 与 flaviagiammarino/path-vqa (test 6,719), 字段 image / question / answer。
首次运行需联网 (服务器上 HF_ENDPOINT=https://hf-mirror.com), 之后可 HF_HUB_OFFLINE=1。

用法:
  python eval/eval_medvqa.py --dataset vqa-rad --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline_server
  python eval/eval_medvqa.py --dataset path-vqa --model ... --adapter outputs/xxx --tag sft_mix_300
  python eval/eval_medvqa.py --dataset path-vqa --n 2000 ...      # 固定种子抽样, 0 = 全部
远程模式同其他脚本: 设 MEDVLM_API_BASE (见 docs/API.md)。

输出: outputs/eval/<vqarad|pathvqa>_<tag>.json 与 _preds.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.prompts import slake_prompt  # noqa: E402

# (仓库, 输出文件前缀); 只下测试集分片 (PathVQA 的训练 + 验证分片约 750 MB, 用不上)
DATASETS = {
    "vqa-rad": ("flaviagiammarino/vqa-rad", "vqarad"),
    "path-vqa": ("flaviagiammarino/path-vqa", "pathvqa"),
}
TEST_FILES = "data/test-*.parquet"


def is_closed(answer: str) -> bool:
    return M.normalize(answer) in {"yes", "no"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n", type=int, default=0, help="固定种子抽 N 题, 0 = 整个测试集")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    repo_id, short = DATASETS[args.dataset]
    ds = load_dataset(repo_id, data_files={"test": TEST_FILES}, split="test")
    if args.n and args.n < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.n))
    print(f"{repo_id} test: {len(ds)} 题")

    model, processor = load_model(args.model, args.adapter, args.load_4bit, max_pixels=args.max_pixels)
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    closed, open_em, open_rec, open_f1 = [], [], [], []
    yn = {"gold_yes": 0, "gold_no": 0, "pred_yes": 0, "pred_no": 0, "pred_other": 0,
          "no_to_yes": 0, "yes_to_no": 0}
    t0 = time.time()
    with open(out_dir / f"{short}_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for i, r in enumerate(tqdm(ds)):
            q, gold = str(r["question"]).strip(), str(r["answer"]).strip()
            c = is_closed(gold)
            pred = generate(model, processor, slake_prompt(q, "CLOSED" if c else "OPEN"),
                            r["image"].convert("RGB"), args.max_new_tokens)
            if c:
                g, p = M.yes_no(gold), M.yes_no(pred)
                score = float(p == g)
                closed.append(score)
                yn["gold_" + g] += 1
                yn["pred_" + p] += 1
                yn["no_to_yes"] += int(g == "no" and p == "yes")
                yn["yes_to_no"] += int(g == "yes" and p == "no")
            else:
                score = M.exact_match(pred, gold)
                open_em.append(score)
                open_rec.append(M.token_recall(pred, gold))
                open_f1.append(M.token_f1(pred, gold))
            fout.write(json.dumps({"i": i, "closed": c, "question": q, "gold": gold, "pred": pred,
                                   "score": score}, ensure_ascii=False) + "\n")

    def mean(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else None

    nc = len(closed)
    summary = {
        "dataset": f"{repo_id}/test", "model": args.model, "adapter": args.adapter, "backend": backend_info(model),
        "n": len(ds), "n_closed": nc, "n_open": len(open_em), "sample_seed": args.seed if args.n else None,
        "seconds": round(time.time() - t0, 1),
        "metrics": {"closed_acc": mean(closed), "open_em": mean(open_em),
                    "open_recall": mean(open_rec), "open_f1": mean(open_f1)},
        "closed_yes_no": {**yn,
                          "gold_yes_pct": round(100 * yn["gold_yes"] / nc, 2) if nc else None,
                          "pred_yes_pct": round(100 * yn["pred_yes"] / nc, 2) if nc else None},
    }
    (out_dir / f"{short}_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
