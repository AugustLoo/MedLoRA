"""第二个通用能力探针: MMBench (英文 dev 集) 固定抽样, 四选一。

为什么要第二个探针: TextVQA 只有 300 题, 而且是短答式 OCR 题, 和微调后的输出格式太像,
会低估遗忘。C2 那条「回放代价」曲线的纵轴现在就架在它上面 —— 形状可信, 幅度不可信。
MMBench 覆盖 20 个能力维度 (属性识别、空间关系、常识推理、图表理解……), 选择题格式与
SLAKE 短答式无关, 更能暴露长程能力的漂移。

与 eval_general.py 同一套约定: 固定种子抽样, 同一批题在基座和每个 adapter 上各跑一次,
差值就是遗忘量; 逐题预测入 outputs/eval/mmbench_<tag>_preds.jsonl, 汇总入 mmbench_<tag>.json。

评分: 只取模型输出里第一个 A-D 字母, 与标准答案比较。不做 CircularEval (选项轮换), 那要 4 倍推理;
作为遗忘探针, 单次 500 题的准确率差值已经够用, 且各模型条件完全相同。

数据: lmms-lab/MMBench, config "en", split "dev" (test 集没有答案)。首次运行需要联网
(服务器上走 HF_ENDPOINT=https://hf-mirror.com), 之后可以 HF_HUB_OFFLINE=1。
若仓库名或 config 名有变, 改下面的 DATASET / CONFIG 两个常量即可。

用法:
  python eval/eval_mmbench.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline_server
  python eval/eval_mmbench.py --model ... --adapter outputs/xxx --tag sft_mix_300
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.model import generate, load_model  # noqa: E402
from medvlm.prompts import MMBENCH  # noqa: E402

DATASET = "lmms-lab/MMBench"
CONFIG = "en"
SPLIT = "dev"
LETTERS = "ABCD"


def build_prompt(r) -> tuple[str, list[str]]:
    """把一道题拼成提示词; 只列非空选项 (MMBench 有些题只有 2-3 个选项)。"""
    opts = []
    for L in LETTERS:
        v = r.get(L)
        if v is None:
            continue
        v = str(v).strip()
        if v and v.lower() != "nan":
            opts.append(f"{L}. {v}")
    hint = str(r.get("hint") or "").strip()
    hint = "" if hint.lower() == "nan" else hint
    return MMBENCH.format(hint=(hint + "\n") if hint else "", question=str(r["question"]).strip(),
                          options="\n".join(opts)), opts


def parse_letter(pred: str, n_opts: int) -> str:
    """取输出里第一个合法的选项字母; 没有就返回空串 (记错)。"""
    valid = LETTERS[:n_opts]
    m = re.search(r"\b([" + valid + r"])\b", pred.upper())
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    args = ap.parse_args()

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    model, processor = load_model(args.model, args.adapter, args.load_4bit, max_pixels=args.max_pixels)

    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, by_cat = [], defaultdict(list)
    with open(out_dir / f"mmbench_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for r in tqdm(ds):
            prompt, opts = build_prompt(r)
            pred = generate(model, processor, prompt, r["image"], 8)
            letter = parse_letter(pred, len(opts))
            gold = str(r["answer"]).strip().upper()
            s = float(letter == gold)
            scores.append(s)
            cat = str(r.get("category") or r.get("l2-category") or "unknown")
            by_cat[cat].append(s)
            fout.write(json.dumps({"index": r.get("index"), "category": cat, "question": r["question"],
                                   "options": opts, "gold": gold, "pred_raw": pred, "pred": letter,
                                   "score": s}, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model, "adapter": args.adapter, "n": len(scores), "seed": args.seed,
        "dataset": f"{DATASET}/{CONFIG}/{SPLIT}",
        "mmbench_acc": round(100 * sum(scores) / len(scores), 2),
        "unparsed": sum(1 for line in open(out_dir / f"mmbench_{args.tag}_preds.jsonl", encoding="utf-8")
                        if json.loads(line)["pred"] == ""),
        "by_category": {k: {"n": len(v), "acc": round(100 * sum(v) / len(v), 2)}
                        for k, v in sorted(by_cat.items())},
    }
    (out_dir / f"mmbench_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))


if __name__ == "__main__":
    main()
