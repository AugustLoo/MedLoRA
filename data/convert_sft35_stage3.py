"""35B 第③步「推理训练」数据: 带解析的医学考题, 统一成「先给答案, 再给解析」(不开思考模式)。

来源 (服务器 /home/ubuntu/chunqian/data/sft35_raw/, scripts/dl_sft35.sh 下载):
    cmexam/train.json                         CMExam 训练集 54,497 题 (中国执业医师考试, 有解析), Apache-2.0
    medmcqa/data/train-*.parquet              MedMCQA 训练集 182,822 题 (印度医学考试, 部分有解析), Apache-2.0
    medical_o1/medical_o1_sft{,_Chinese}.json medical-o1-reasoning-SFT 中英文, 只用最终回答 Response (不要长推理链), Apache-2.0
只用训练划分; 验证 / 测试划分留给评测, 并且训练题与所有评测题比对 (sft35_common.EvalIndex), 重复或改写的剔除。
CMExam / MedMCQA 只要单选题、解析不太短 (≥ 20 字符)、选项完整的; MedMCQA 解析开头的 "Ans. is 'c' i.e., …" 去掉。
输出 (--out, 默认 data/processed/):
    sft35_reasoning.json        [{"messages": [user, assistant], "source", "id"}]
    sft35_reasoning_stats.json

用法 (服务器, 不占卡):
    python data/convert_sft35_stage3.py --tokenizer /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sft35_common import RAW, REPO, build_eval_index, fits, has_pii, load_any, sample, strip_html, token_counter

sys.path.insert(0, str(REPO))
from medvlm.prompts import MCQ_EXPLAIN_EN, MCQ_EXPLAIN_ZH  # noqa: E402

_ANS_PREFIX = re.compile(r"^\s*(ans(wer)?\.?\s*(is)?\s*[:\-]?\s*['\"‘’(]?\s*[a-d]\s*['\"‘’)]?\s*[,.:;-]?\s*(i\.?\s*e\.?\s*[,:]?)?)",
                         re.I)


def cmexam_row(r: dict) -> tuple[dict | None, str]:
    opts = {str(o["key"]).strip(): str(o["value"]).strip() for o in (r.get("Options") or []) if o.get("key")}
    ans = str(r.get("Answer", "")).strip().upper()
    exp = strip_html(r.get("Explanation") or "")
    if len(ans) != 1 or ans not in opts:
        return None, "不是单选题或答案不在选项里"
    if len(opts) < 4 or any(not v for v in opts.values()):
        return None, "选项不完整"
    if len(exp) < 20:
        return None, "没有解析或解析太短"
    q = str(r["Question"]).strip()
    user = MCQ_EXPLAIN_ZH.format(question=q, options="\n".join(f"{k}. {v}" for k, v in sorted(opts.items())))
    return {"q": q, "messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": f"答案：{ans}\n解析：{exp}"}]}, ""


def medmcqa_row(r: dict) -> tuple[dict | None, str]:
    if str(r.get("choice_type", "")).lower() != "single":
        return None, "不是单选题"
    opts = {k: str(r.get(f"op{k.lower()}") or "").strip() for k in "ABCD"}
    if any(not v for v in opts.values()):
        return None, "选项不完整"
    try:
        ans = "ABCD"[int(r["cop"])]
    except (TypeError, ValueError, IndexError):
        return None, "答案编号不对"
    exp = _ANS_PREFIX.sub("", strip_html(r.get("exp") or "")).strip()
    if len(exp) < 20:
        return None, "没有解析或解析太短"
    q = str(r["question"]).strip()
    user = MCQ_EXPLAIN_EN.format(question=q, options="\n".join(f"{k}. {v}" for k, v in opts.items()))
    return {"q": q, "messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": f"Answer: {ans}\nExplanation: {exp}"}]}, ""


def o1_row(r: dict) -> tuple[dict | None, str]:
    q, resp = strip_html(r.get("Question") or ""), strip_html(r.get("Response") or "")
    if len(q) < 10 or len(resp) < 20:
        return None, "题目或回答太短"
    return {"q": q, "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": resp}]}, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="35B 模型目录 (计 token 用); 不给就按 字符数/2 粗估")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(REPO / "data" / "processed"))
    ap.add_argument("--cmexam", type=int, default=5000)
    ap.add_argument("--medmcqa", type=int, default=5000)
    ap.add_argument("--o1-zh", type=int, default=1500)
    ap.add_argument("--o1-en", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample", type=float, default=3.0)
    args = ap.parse_args()
    raw = Path(args.raw)
    count = token_counter(args.tokenizer)
    print("建评测题索引")
    idx = build_eval_index()

    import pandas as pd

    medmcqa = pd.read_parquet(raw / "medmcqa" / "data" / "train-00000-of-00001.parquet").to_dict("records")
    sources = [
        ("cmexam", args.cmexam, load_any(raw / "cmexam" / "train.json"), cmexam_row),
        ("medmcqa", args.medmcqa, medmcqa, medmcqa_row),
        ("medical_o1_zh", args.o1_zh, load_any(raw / "medical_o1" / "medical_o1_sft_Chinese.json"), o1_row),
        ("medical_o1_en", args.o1_en, load_any(raw / "medical_o1" / "medical_o1_sft.json"), o1_row),
    ]
    out, stats = [], {}
    for k, (name, target, rows, conv) in enumerate(sources):
        pool = sample(rows, int(target * args.oversample), args.seed + k)
        kept, drop = [], Counter()
        for i, r in enumerate(pool):
            ex, why = conv(r)
            if ex is None:
                drop[why] += 1
                continue
            if has_pii(ex["messages"][1]["content"]):
                drop["回答含联系方式 / 网址"] += 1
                continue
            if idx.contains(ex["q"]):
                drop["与评测题重复或改写"] += 1
                continue
            if not fits(ex["messages"], count, args.max_tokens):
                drop["超长"] += 1
                continue
            kept.append({"messages": ex["messages"], "source": name, "id": str(r.get("id", f"{name}-{i}"))})
            if len(kept) >= target:
                break
        stats[name] = {"total": len(rows), "pool": len(pool), "kept": len(kept), "target": target, "dropped": dict(drop)}
        print(name, json.dumps(stats[name], ensure_ascii=False), flush=True)
        if len(kept) < target:
            print(f"!! {name} 只收到 {len(kept)} / {target} 条, 可调大 --oversample", flush=True)
        out += kept
    stats["_eval_index"] = idx.sources
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "sft35_reasoning.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (o / "sft35_reasoning_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("total", len(out))


if __name__ == "__main__":
    main()
