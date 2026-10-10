"""用 CheXbert 判「描述里有没有说出片子上的病」, 代替 eval_chexpert.py 的关键词判断 (week4 下周计划第一项)。

CheXbert (Smit et al., 2020) 读一段放射科文字, 给 14 类病变各打一个标签: 空 / 阳性 / 阴性 / 不确定 (No Finding 只有空 / 阳性)。
CheXpert Plus 的标准答案就是它从医生报告里读出来的, 现在用同一个模型读 35B 写的描述, 两边逐项比 —— 同一把尺子量两边。

模型: StanfordAIMI/RRG_scorers 的 chexbert.pth + bert-base-uncased 的配置和词表, 结构照 f1chexbert 0.0.2
(BERT + 13 个 4 类头 + 1 个 2 类头, 取 [CLS]), 不装那个包以免改动环境里的库版本。只用 CPU (容器看到的卡是主机 6、7 号, 不能用)。

两步:
  python eval/chexbert_judge.py validate                 # 尺子校验: 读测试片的医生报告, 和表格里现成的标签比一致率
  python eval/chexbert_judge.py score interns2_base ...   # 读各模型 chexpert_<tag>_preds.jsonl 里的描述, 逐项比
逐条标签含由报告派生的信息, 只留服务器 (outputs/eval/chexbert_<tag>_labels.jsonl); 汇总 chexbert_<tag>.json 只有比例, 可以带回本机。
运行位置: user0 容器, chunqian 环境。
"""
from __future__ import annotations

import os

# 线程数必须在 import torch 之前限住: 这台服务器 128 核, 默认开上百个线程互相抢, 反而慢几百倍 (docs/SERVER.md)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
PAIRS = Path("/workspace/rafi/chexpert_plus_subset/pairs_with_text_labels.csv")
MODEL_DIR = Path(os.environ.get("CHEXBERT_DIR", "/workspace/chunqian/models/chexbert"))
OUT = REPO / "outputs" / "eval"
CONDITIONS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices", "No Finding"]
FINDINGS = CONDITIONS[:12]  # 判断漏报只看这 12 类病变 (不算 Support Devices 与 No Finding), 与 eval_chexpert.py 相同
CLASS = {0: "blank", 1: "positive", 2: "negative", 3: "uncertain"}  # CheXbert 输出类别 (No Finding 头只有 0 / 1)


class Labeler(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_DIR / "bert-base-uncased"))
        hidden = self.bert.config.hidden_size
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)] + [nn.Linear(hidden, 2)])

    def forward(self, ids, mask):
        cls = self.bert(ids, attention_mask=mask)[0][:, 0, :]
        return [h(cls) for h in self.linear_heads]


def load_labeler():
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(MODEL_DIR / "bert-base-uncased")
    model = Labeler()
    state = torch.load(MODEL_DIR / "chexbert.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # 只容许新旧 transformers 之间的无害差别 (position_ids 缓冲区); 其余缺键说明结构没对上, 结果会是乱的
    bad = [k for k in list(missing) + list(unexpected) if "position_ids" not in k]
    if bad:
        raise SystemExit(f"权重没对上: {bad[:8]}")
    return tok, model.eval()


@torch.inference_mode()
def label_texts(texts, tok, model, batch=16, log_every=20):
    """每段文字 → {病变名: blank/positive/negative/uncertain}。切分方式照 f1chexbert: 最长 512 个 token。
    按长度排序后分批 (少补齐、CPU 上快不少), 结果按原顺序返回; 每 log_every 批打印一次进度。"""
    enc = [tok.encode(" ".join(str(t).split()) or "[PAD]", truncation=True, max_length=512) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [None] * len(enc)
    t0 = time.time()
    for b, s in enumerate(range(0, len(order), batch)):
        idx = order[s:s + batch]
        n = max(len(enc[i]) for i in idx)
        ids = torch.tensor([enc[i] + [0] * (n - len(enc[i])) for i in idx])
        heads = model(ids, (ids != 0).long())
        preds = [h.argmax(dim=1).tolist() for h in heads]
        for j, i in enumerate(idx):
            out[i] = {c: CLASS[preds[k][j]] for k, c in enumerate(CONDITIONS)}
        if (b + 1) % log_every == 0:
            print(f"  {s + len(idx)}/{len(enc)}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return out


def test_rows():
    rows = [r for r in csv.DictReader(open(PAIRS, encoding="utf-8")) if r.get("our_split") == "test"]
    return rows  # 顺序与 eval_chexpert.py 相同, preds 里的 i 就是这里的下标


def gold_class(v: str) -> str:
    v = (v or "").strip()
    return {"": "blank", "1.0": "positive", "0.0": "negative", "-1.0": "uncertain"}.get(v, "blank")


def cmd_validate(args):
    rows = test_rows()
    if args.limit:
        rows = rows[: args.limit]
    tok, model = load_labeler()
    t0 = time.time()
    labels = label_texts([r["report_text"] for r in rows], tok, model)
    agree, total, pos_hit, pos_n = Counter(), Counter(), Counter(), Counter()
    for r, lab in zip(rows, labels):
        for c in CONDITIONS:
            g = gold_class(r.get(c))
            total[c] += 1
            agree[c] += g == lab[c]
            if g == "positive":
                pos_n[c] += 1
                pos_hit[c] += lab[c] == "positive"
    res = {c: {"agree_pct": round(100 * agree[c] / total[c], 2),
               "gold_positive": pos_n[c],
               "positive_recovered_pct": round(100 * pos_hit[c] / pos_n[c], 2) if pos_n[c] else None} for c in CONDITIONS}
    overall = round(100 * sum(agree.values()) / sum(total.values()), 2)
    summary = {"what": "CheXbert re-labelling of the test-split report_text vs the labels in the table",
               "n_reports": len(rows), "seconds": round(time.time() - t0, 1), "overall_agree_pct": overall, "per_condition": res}
    (OUT / "chexbert_validate.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_score(args):
    rows = test_rows()
    tok, model = load_labeler()
    for tag in args.tags:
        preds = [json.loads(x) for x in open(OUT / f"chexpert_{tag}_preds.jsonl", encoding="utf-8")]
        t0 = time.time()
        labels = label_texts([p["desc"] for p in preds], tok, model)
        g = defaultdict(Counter)
        per = {f: Counter() for f in FINDINGS}
        with open(OUT / f"chexbert_{tag}_labels.jsonl", "w", encoding="utf-8") as fout:  # 只留服务器
            for p, lab in zip(preds, labels):
                r = rows[p["i"]]
                said = [f for f in FINDINGS if lab[f] in ("positive", "uncertain")]
                G = g[p["group"]]
                G["n"] += 1
                G["desc_reports_any"] += bool(said)
                G["closed_yes"] += p["closed"] == "yes"
                G["closed_yes_desc_none"] += p["closed"] == "yes" and not said
                G["keyword_reports_any"] += bool(p["reports_any"])
                G["agree_with_keyword"] += bool(said) == bool(p["reports_any"])
                for f in FINDINGS:
                    if gold_class(r.get(f)) == "positive":
                        per[f]["n"] += 1
                        per[f]["named_positive"] += lab[f] == "positive"
                        per[f]["named_pos_or_unc"] += lab[f] in ("positive", "uncertain")
                        per[f]["called_negative"] += lab[f] == "negative"
                fout.write(json.dumps({"i": p["i"], "group": p["group"], "chexbert": lab}, ensure_ascii=False) + "\n")
        pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
        A, N = g["abnormal"], g["normal"]
        summary = {
            "tag": tag, "judge": "CheXbert (StanfordAIMI/RRG_scorers chexbert.pth) on the model descriptions",
            "seconds": round(time.time() - t0, 1), "groups": {k: v["n"] for k, v in g.items()},
            "abnormal_no_finding_reported_pct": pct(A["n"] - A["desc_reports_any"], A["n"]),
            "normal_finding_reported_pct": pct(N["desc_reports_any"], N["n"]),
            "abnormal_closed_yes_but_desc_none_pct": pct(A["closed_yes_desc_none"], A["closed_yes"]),
            "keyword_judge_abnormal_no_finding_pct": pct(A["n"] - A["keyword_reports_any"], A["n"]),
            "agreement_with_keyword_judge_pct": pct(sum(v["agree_with_keyword"] for v in g.values()),
                                                     sum(v["n"] for v in g.values())),
            "per_finding": {f: {"n": c["n"], "named_positive_pct": pct(c["named_positive"], c["n"]),
                                "named_pos_or_uncertain_pct": pct(c["named_pos_or_unc"], c["n"]),
                                "called_negative_pct": pct(c["called_negative"], c["n"])}
                            for f, c in per.items() if c["n"]},
        }
        (OUT / f"chexbert_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "per_finding"}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("tags", nargs="+")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    {"validate": cmd_validate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
