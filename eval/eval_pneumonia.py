"""「漏报异常」探针: PneumoniaMNIST 官方测试集 624 张儿童胸片 (390 肺炎 / 234 正常), 只测不训。

为什么: 第 3 周在 SLAKE 上试过「把异常片子说成正常」, 但只有 44 张异常图, 同一条件两个种子 13 vs 3, 结论撤回。
这里每张图都有明确的正常 / 肺炎标签, 数量大一个量级。两种问法都跑:
  1. 是非题: "Does this chest X-ray show pneumonia? Answer with yes or no only." (与 SLAKE 封闭题同一句收尾)
     → 灵敏度 (肺炎里答 yes 的比例)、特异度、准确率、漏报数 (肺炎答 no)、误报数、预测 yes 比例、答不出 yes/no 的数量
  2. 自由描述 (--freetext): "Describe this chest X-ray in two or three sentences, including any abnormal findings."
     → 描述里有没有报告异常: 先去掉否定说法 ("no evidence of ...", "without ..."), 再看剩下的文字里有没有异常词。
       分两种口径: 任何异常 (含心脏增大) 与肺部异常 (阴影、实变、浸润、积液、肺炎……)。另统计「是非题答 yes、描述却没报肺部异常」。
     2026-10-01 改: 最初用 eval_openended.says_normal (找「说正常」的措辞), 人工抽查发现它漏掉
     "no apparent abnormalities"、"absence of major pulmonary abnormalities" 等说法, 而且各模型漏掉的条数不同 (21-99 条),
     会扭曲模型间的比较; 改为「有没有报告异常」后抽查 50 条全对。says_normal 仍逐条保存, 便于追溯。
     已有的逐题记录可用 scripts/rescore_pneumonia.py 按新口径重算, 不必重跑模型。
局限: 儿童胸片, 224×224, 与 SLAKE 的成人影像不同; 绝对分数不宜与别的数据集直接比, 主要看训练前后的变化。

数据: MedMNIST v2 PneumoniaMNIST (CC BY 4.0) 的 224 分辨率版本。只需测试集:
  data/raw/medmnist/pneumoniamnist_224_test.npz (键 test_images [624,224,224] uint8, test_labels [624], 1 = 肺炎)
  由完整的 pneumoniamnist_224.npz 抽出 (键名相同, 本脚本两种文件都能读)。

用法:
  python eval/eval_pneumonia.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline_server --freetext
  远程模式同其他脚本: 设 MEDVLM_API_BASE (见 docs/API.md)。
输出: outputs/eval/pneumonia_<tag>.json 与 _preds.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.prompts import SLAKE_CLOSED  # noqa: E402

sys.path.insert(0, str(REPO / "eval"))
from eval_openended import ABNORMAL_WORD, NEGATION, says_normal  # noqa: E402

DEFAULT_NPZ = REPO / "data" / "raw" / "medmnist" / "pneumoniamnist_224_test.npz"
Q_CLOSED = SLAKE_CLOSED.format(question="Does this chest X-ray show pneumonia?")
Q_FREE = "Describe this chest X-ray in two or three sentences, including any abnormal findings."
LUNG_WORD = re.compile(r"opacit|consolidat|infiltrat|pneumon|effusion|atelecta|ground-glass|pneumothorax|nodul|mass|"
                       r"infection|inflammat|fluid accumulation|edema|oedema", re.I)


def reports_any_abnormality(text: str) -> bool:
    """去掉否定说法后, 还提到任何异常词 (含心脏增大等非肺部异常)。"""
    return bool(ABNORMAL_WORD.search(NEGATION.sub(" ", text)))


def reports_lung_abnormality(text: str) -> bool:
    """去掉否定说法后, 还提到肺部异常词。肺炎片描述里没有它, 就算漏报。"""
    return bool(LUNG_WORD.search(NEGATION.sub(" ", text)))


def freetext_summary(rows: list[dict]) -> dict:
    """由逐题记录 (含 label / closed / desc) 计算自由描述指标; rescore_pneumonia.py 也调用它。"""
    P = [r for r in rows if r["label"] == "pneumonia"]
    N = [r for r in rows if r["label"] == "normal"]
    pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    no_lung_p = sum(not reports_lung_abnormality(r["desc"]) for r in P)
    no_lung_n = sum(not reports_lung_abnormality(r["desc"]) for r in N)
    no_any_p = sum(not reports_any_abnormality(r["desc"]) for r in P)
    yes_p = [r for r in P if r["closed"] == "yes"]
    contra = sum(not reports_lung_abnormality(r["desc"]) for r in yes_p)
    return {"pneu_no_lung_abnormality": no_lung_p, "pneu_no_lung_abnormality_pct": pct(no_lung_p, len(P)),
            "normal_no_lung_abnormality": no_lung_n, "normal_no_lung_abnormality_pct": pct(no_lung_n, len(N)),
            "lung_discrimination_pts": round(pct(no_lung_n, len(N)) - pct(no_lung_p, len(P)), 2) if P and N else None,
            "pneu_no_any_abnormality": no_any_p, "pneu_no_any_abnormality_pct": pct(no_any_p, len(P)),
            "closed_yes_but_desc_no_lung": contra, "closed_yes_pneu": len(yes_p),
            "closed_yes_but_desc_no_lung_pct": pct(contra, len(yes_p)),
            "legacy_says_normal_pneu": sum(says_normal(r["desc"]) for r in P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--freetext", action="store_true", help="另外跑自由描述 (每张多一次生成)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz)
    x, y = d["test_images"], d["test_labels"].reshape(-1).astype(int)
    if args.limit:
        x, y = x[: args.limit], y[: args.limit]
    print(f"PneumoniaMNIST test: {len(y)} 张 (肺炎 {int(y.sum())} / 正常 {int((1 - y).sum())})")

    model, processor = load_model(args.model, args.adapter, args.load_4bit)
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    c = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "unparsed": 0}
    rows = []
    t0 = time.time()
    with open(out_dir / f"pneumonia_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(y))):
            img = Image.fromarray(x[i]).convert("RGB")
            ans = generate(model, processor, Q_CLOSED, img, 8)
            p = M.yes_no(ans)
            pneu = bool(y[i])
            if p == "other":
                c["unparsed"] += 1
            if pneu:
                c["tp" if p == "yes" else "fn"] += 1   # 答不出 yes/no 的肺炎片按漏报计
            elif p != "other":
                c["tn" if p == "no" else "fp"] += 1    # 正常片答不出的不算误报, 只在特异度里算错
            row = {"i": i, "label": "pneumonia" if pneu else "normal", "closed_raw": ans, "closed": p}
            if args.freetext:
                desc = generate(model, processor, Q_FREE, img, 128)
                row.update({"desc": desc, "says_normal": says_normal(desc),
                            "reports_lung": reports_lung_abnormality(desc), "reports_any": reports_any_abnormality(desc)})
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_p, n_n = int(y.sum()), int((1 - y).sum())
    pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    summary = {
        "dataset": "PneumoniaMNIST 224 test", "model": args.model, "adapter": args.adapter,
        "backend": backend_info(model), "n": len(y), "n_pneumonia": n_p, "n_normal": n_n,
        "seconds": round(time.time() - t0, 1),
        "closed": {**c, "accuracy": pct(c["tp"] + c["tn"], len(y)), "sensitivity": pct(c["tp"], n_p),
                   "specificity": pct(c["tn"], n_n), "missed_pneumonia": c["fn"],
                   "pred_yes_pct": pct(c["tp"] + c["fp"], len(y))},
    }
    if args.freetext:
        summary["freetext"] = freetext_summary(rows)
    (out_dir / f"pneumonia_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
