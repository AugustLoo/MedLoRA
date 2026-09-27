"""开放式生成探针: 测「写一段话」的能力有没有被短答式微调带偏。

为什么要它: 另外三个通用探针的输出都很短 (TextVQA 一两个词, MMBench 一个字母, PubMedQA 一个词)。
短答式 SFT 最可能带偏的恰恰是「写长一点」—— 例如让它描述一张图, 它只回一个词, 或者开始说 unanswerable。
这两种退化, 短输出的探针都看不见。

两个任务, 同一套约定 (固定抽样、贪心解码、逐条预测入 outputs/eval/):

  coco   通用图片描述。lmms-lab/COCO-Caption2017 val 固定抽 300 张 (seed 42), 每张 5 条人工参考描述。
         指标: CIDEr / ROUGE-L (与参考描述的相似度)、平均词数、少于 5 个词的比例 (「缩成短答」)、
               拒答比例 (unanswerable / cannot / not possible ...)、重复度 (distinct-2)。
  slake  医学图片描述。SLAKE 测试集全部 96 张图, 每张图的成像方式和身体部位是标注好的, 异常与否从同一张图的
         封闭题答案推出 (「Is the lung healthy? No」「Are there abnormalities? Yes」...)。不需要参考描述:
         指标: 成像方式说对的比例、身体部位说对的比例、异常图被说成「正常」的比例 (B2 的失败方式)、
               平均词数、缩成短答比例、拒答比例。

用法 (与其他评估脚本相同, 支持本地模型和远程接口):
  python eval/eval_openended.py --model <模型或服务端模型名> --tag interns2_base
  python eval/eval_openended.py --task slake --limit 5 ...        # 只跑医学描述前 5 张 (调试)
首次跑 coco 需要联网 (服务器上 HF_ENDPOINT=https://hf-mirror.com), 只下 val 的 815 MB; 抽好的 300 张存到
$HF_HOME/medlora_coco_val_n300_seed42, 之后离线直接读。
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.slake import RAW  # noqa: E402

COCO_DATASET = "lmms-lab/COCO-Caption2017"
PROMPT_COCO = "Describe this image in two or three sentences."
PROMPT_SLAKE = ("Describe this medical image in two or three sentences: "
                "the imaging modality, the body region, and any abnormal findings.")

REFUSAL = re.compile(r"\b(unanswerable|cannot|can't|unable|not possible|no image|sorry)\b", re.I)
MODALITY = {"CT": r"\bct\b|computed tomograph", "MRI": r"\bmri\b|magnetic resonance|\bt1\b|\bt2\b|flair",
            "X-Ray": r"x-?ray|radiograph"}
LOCATION = {"Lung": r"lung|chest|thora|pulmonar", "Abdomen": r"abdom|liver|kidney|spleen|bowel|pancrea",
            "Brain_Tissue": r"brain|head|cran|skull|cerebr", "Brain": r"brain|head|cran|skull|cerebr",
            "Brain_Face": r"brain|head|face|facial|orbit|skull|sinus", "Pelvic Cavity": r"pelvi|bladder|rect|uter|prostat",
            "Neck": r"neck|thyroid|cervical|trache", "Chest_mediastinal": r"chest|mediastin|heart|thora"}
NORMAL_CLAIM = re.compile(r"\bno (?:acute|obvious|significant|visible|evident)?\s*(?:abnormal|lesion|finding)|"
                          r"\bnormal\b|unremarkable|within normal limits", re.I)
ABNORMAL_WORD = re.compile(r"abnormal|lesion|mass|tumou?r|nodul|opacit|effusion|consolidat|infiltrat|edema|oedema|"
                           r"hemorrhag|haemorrhag|fractur|cancer|carcinom|enlarg|cardiomegal|pneumon|atelecta|"
                           r"pneumothorax|cyst|metasta", re.I)


# ---------------------------------------------------------------- 文本指标
def says_normal(text: str) -> bool:
    """描述里宣称正常, 且去掉「no ... abnormal」这类否定说法后不再提任何异常词。"""
    if not NORMAL_CLAIM.search(text):
        return False
    return not ABNORMAL_WORD.search(NORMAL_CLAIM.sub(" ", text))


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_l(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    dp = [0] * (len(ref) + 1)
    for p in pred:
        prev = 0
        for j, r in enumerate(ref, 1):
            cur = dp[j]
            dp[j] = prev + 1 if p == r else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    prec, rec = lcs / len(pred), lcs / len(ref)
    return 0.0 if lcs == 0 else (1 + 1.2 ** 2) * prec * rec / (rec + 1.2 ** 2 * prec)


def ngrams(t: list[str], n: int) -> Counter:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def cider_d(preds: list[list[str]], refs: list[list[list[str]]], sigma: float = 6.0) -> list[float]:
    """CIDEr-D (Vedantam et al. 2015), 文档频率取自本次评估的参考描述集合, 与 pycocoevalcap 的做法一致。"""
    N = len(refs)
    df = [Counter() for _ in range(4)]
    for rs in refs:
        for n in range(4):
            seen = set()
            for r in rs:
                seen |= set(ngrams(r, n + 1))
            df[n].update(seen)
    log_n = math.log(float(N))

    def vec(t):
        out, norm = [], []
        for n in range(4):
            c = ngrams(t, n + 1)
            v = {g: tf * (log_n - math.log(max(1.0, df[n][g]))) for g, tf in c.items()}
            out.append(v)
            norm.append(math.sqrt(sum(x * x for x in v.values())))
        return out, norm, len(t)

    scores = []
    for p, rs in zip(preds, refs):
        vp, np_, lp = vec(p)
        s = [0.0] * 4
        for r in rs:
            vr, nr, lr = vec(r)
            for n in range(4):
                dot = sum(min(vp[n][g], vr[n].get(g, 0.0)) * vr[n].get(g, 0.0) for g in vp[n])
                if np_[n] and nr[n]:
                    s[n] += dot / (np_[n] * nr[n]) * math.exp(-((lp - lr) ** 2) / (2 * sigma ** 2))
        scores.append(10.0 * sum(x / len(rs) for x in s) / 4)
    return scores


def distinct2(t: list[str]) -> float:
    grams = list(zip(t, t[1:]))
    return len(set(grams)) / len(grams) if grams else 1.0


def style_stats(texts: list[str]) -> dict:
    tk = [toks(t) for t in texts]
    n = len(texts)
    return {"mean_words": round(sum(map(len, tk)) / n, 1),
            "short_lt5_pct": round(100 * sum(len(t) < 5 for t in tk) / n, 1),
            "refusal_pct": round(100 * sum(bool(REFUSAL.search(t)) for t in texts) / n, 1),
            "distinct2": round(sum(distinct2(t) for t in tk) / n, 3)}


# ---------------------------------------------------------------- 数据
def load_coco(n: int, seed: int):
    """只下载 val 划分 (两个 parquet 共 815 MB, 不下 6.6 GB 的 test); 抽好的 n 张存成本地文件,
    之后离线直接读这份文件, 不依赖 datasets 的离线缓存机制 (主机上吃过这个亏, 见 docs/INTERNS2.md)。"""
    import os
    from datasets import load_dataset, load_from_disk
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local = hf_home / f"medlora_coco_val_n{n}_seed{seed}"
    if local.exists():
        ds = load_from_disk(str(local))
        print(f"[coco] 读本地抽样 {local} ({len(ds)} 张)")
    else:
        full = load_dataset(COCO_DATASET, data_files={"val": "data/val-*.parquet"}, split="val")
        print(f"[coco] val {len(full)} 张, 列 {full.column_names}")
        ds = full.shuffle(seed=seed).select(range(min(n, len(full))))
        ds.save_to_disk(str(local))
        print(f"[coco] 抽样已存到 {local}")
    cap_col = next(c for c in ("answer", "captions", "caption", "sentences") if c in ds.column_names)
    for r in ds:
        caps = r[cap_col]
        caps = [caps] if isinstance(caps, str) else list(caps)
        yield {"id": r.get("question_id") or r.get("id") or r.get("file_name"), "image": r["image"], "refs": caps}


def load_slake_images():
    rows = json.loads(next(RAW.rglob("test.json")).read_text(encoding="utf-8"))
    img_root = next(RAW.rglob("imgs"))
    by = defaultdict(list)
    for r in rows:
        if str(r.get("q_lang", "en")).lower() == "en":
            by[r["img_name"]].append(r)
    for name, qs in sorted(by.items()):
        abnormal = None
        for q in qs:
            ql, a = q["question"].lower(), str(q["answer"]).strip().lower()
            if q["answer_type"] != "CLOSED" or a not in ("yes", "no"):
                continue
            if "healthy" in ql or re.search(r"\bnormal\b", ql):
                abnormal = (a == "no") if abnormal is None else (abnormal or a == "no")
            elif "abnormal" in ql:
                abnormal = (a == "yes") if abnormal is None else (abnormal or a == "yes")
        yield {"id": name, "image_path": str(img_root / name), "modality": qs[0]["modality"],
               "location": qs[0]["location"], "abnormal": abnormal}


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--task", choices=["all", "coco", "slake"], default="all")
    ap.add_argument("--n", type=int, default=300, help="COCO 抽多少张")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="每个任务只跑前 N 条 (调试)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    args = ap.parse_args()

    from PIL import Image
    model, processor = load_model(args.model, args.adapter, args.load_4bit, max_pixels=args.max_pixels)
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "adapter": args.adapter, "backend": backend_info(model),
               "max_new_tokens": args.max_new_tokens}

    if args.task in ("all", "coco"):
        items = list(load_coco(args.n, args.seed))[: args.limit or None]
        preds = []
        with open(out_dir / f"openended_coco_{args.tag}_preds.jsonl", "w", encoding="utf-8") as f:
            for it in tqdm(items, desc="coco"):
                p = generate(model, processor, PROMPT_COCO, it["image"], args.max_new_tokens)
                preds.append(p)
                f.write(json.dumps({"id": it["id"], "pred": p, "refs": it["refs"]}, ensure_ascii=False) + "\n")
        cid = cider_d([toks(p) for p in preds], [[toks(r) for r in it["refs"]] for it in items])
        rl = [max(rouge_l(toks(p), toks(r)) for r in it["refs"]) for p, it in zip(preds, items)]
        summary["coco"] = {"n": len(items), "dataset": COCO_DATASET, "prompt": PROMPT_COCO,
                           "cider": round(100 * sum(cid) / len(cid), 2), "rouge_l": round(100 * sum(rl) / len(rl), 2),
                           **style_stats(preds)}

    if args.task in ("all", "slake"):
        items = list(load_slake_images())[: args.limit or None]
        rec = []
        with open(out_dir / f"openended_slake_{args.tag}_preds.jsonl", "w", encoding="utf-8") as f:
            for it in tqdm(items, desc="slake"):
                p = generate(model, processor, PROMPT_SLAKE, Image.open(it["image_path"]), args.max_new_tokens)
                mod_ok = bool(re.search(MODALITY[it["modality"]], p, re.I))
                loc_ok = bool(re.search(LOCATION.get(it["location"], re.escape(it["location"])), p, re.I))
                is_normal = says_normal(p)
                rec.append((it, p, mod_ok, loc_ok, is_normal))
                f.write(json.dumps({"id": it["id"], "modality": it["modality"], "location": it["location"],
                                    "abnormal": it["abnormal"], "pred": p, "modality_ok": mod_ok,
                                    "location_ok": loc_ok, "says_normal": is_normal}, ensure_ascii=False) + "\n")
        ab = [x for x in rec if x[0]["abnormal"] is True]
        summary["slake"] = {"n": len(rec), "prompt": PROMPT_SLAKE,
                            "modality_ok_pct": round(100 * sum(x[2] for x in rec) / len(rec), 1),
                            "location_ok_pct": round(100 * sum(x[3] for x in rec) / len(rec), 1),
                            "n_abnormal": len(ab),
                            "abnormal_called_normal": sum(x[4] for x in ab),
                            **style_stats([x[1] for x in rec])}

    fn = out_dir / f"openended_{args.tag}.json"
    fn.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
