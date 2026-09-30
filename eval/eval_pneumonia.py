"""「漏报异常」探针: PneumoniaMNIST 官方测试集 624 张儿童胸片 (390 肺炎 / 234 正常), 只测不训。

为什么: 第 3 周在 SLAKE 上试过「把异常片子说成正常」, 但只有 44 张异常图, 同一条件两个种子 13 vs 3, 结论撤回。
这里每张图都有明确的正常 / 肺炎标签, 数量大一个量级。两种问法都跑:
  1. 是非题: "Does this chest X-ray show pneumonia? Answer with yes or no only." (与 SLAKE 封闭题同一句收尾)
     → 灵敏度 (肺炎里答 yes 的比例)、特异度、准确率、漏报数 (肺炎答 no)、误报数、预测 yes 比例、答不出 yes/no 的数量
  2. 自由描述 (--freetext): "Describe this chest X-ray in two or three sentences, including any abnormal findings."
     → 肺炎片被描述为正常的数量与比例 (判断沿用 eval_openended.says_normal: 先去掉 "no evidence of ..." 这类否定说法)
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
from eval_openended import says_normal  # noqa: E402

DEFAULT_NPZ = REPO / "data" / "raw" / "medmnist" / "pneumoniamnist_224_test.npz"
Q_CLOSED = SLAKE_CLOSED.format(question="Does this chest X-ray show pneumonia?")
Q_FREE = "Describe this chest X-ray in two or three sentences, including any abnormal findings."


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
    f = {"pneu_called_normal": 0, "normal_called_normal": 0}
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
                sn = says_normal(desc)
                f["pneu_called_normal" if pneu else "normal_called_normal"] += int(sn)
                row.update({"desc": desc, "says_normal": sn})
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
        summary["freetext"] = {**f, "pneu_called_normal_pct": pct(f["pneu_called_normal"], n_p),
                               "normal_called_normal_pct": pct(f["normal_called_normal"], n_n)}
    (out_dir / f"pneumonia_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
