"""短答微调的三个对照实验用的训练数据 (执行方案第 1–2 周)。

都从已有的 slake_train.json (sharegpt 格式, 4,919 条) 出发, 每次只改一样, 条目顺序和图片路径不动:
  R0  nofmt          去掉每道题后面的格式提示句 (现有训练一直带着这句)
  R2  rw-requests    抽 20% 的题, 列出要改写的 (题目, 答案); 改写由 scripts/gen_control_text.py 调原版模型做
      rw-apply       把通过检查的改写放回去 (提示句同时换成「用一句完整的话回答」)
  R1  desc-requests  每张训练图一条, 列出要原版模型描述的图片
      desc-apply     把描述做成训练样本, 单独存一个文件, 训练时和 slake_train.json 一起给
      sample         从别的训练文件 (如服务器上的胸片报告) 里不重复地抽 N 条, 单独存一个文件

用法见 docs/INTERNS2.md「三个对照实验的数据」。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import string
import sys
from pathlib import Path

FORMAT_LINES = ("\nAnswer with yes or no only.", "\nAnswer with a single word or short phrase.")  # medvlm/prompts.py
SENTENCE_LINE = "\nAnswer in one complete sentence."
DESCRIBE = ("Describe this medical image in two or three sentences: "
            "the imaging modality, the body region, and any abnormal findings.")   # 与 eval/eval_openended.py 的 PROMPT_SLAKE 相同
REWRITE = ('Question: {q}\nShort answer: {a}\n'
           'Rewrite the short answer as one complete sentence that answers the question. '
           'Keep the words "{a}" in the sentence and do not add any other medical information. '
           'Output only the sentence.')


def _norm(t: str) -> str:
    t = str(t).lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", t).strip()


def question_of(sample: dict) -> str:
    """题目原文 (含 <image>), 不含后面的格式提示句。"""
    user = sample["messages"][0]["content"]
    for line in FORMAT_LINES:
        if user.endswith(line):
            return user[: -len(line)]
    raise ValueError(f"这条样本的题目后面没有格式提示句: {user[-80:]!r}")


def strip_format(sample: dict) -> dict:
    out = copy.deepcopy(sample)
    out["messages"][0]["content"] = question_of(sample)
    return out


def pick(n_total: int, frac: float, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), round(n_total * frac)))


def rewrite_requests(samples: list[dict], frac: float, seed: int) -> list[dict]:
    out = []
    for i in pick(len(samples), frac, seed):
        q, a = question_of(samples[i]).replace("<image>", "", 1), samples[i]["messages"][1]["content"]
        out.append({"idx": i, "question": q, "answer": a, "prompt": REWRITE.format(q=q, a=a)})
    return out


def rewrite_ok(answer: str, sentence: str) -> bool:
    """一句话, 4–40 个词, 原答案还在: 是 / 否题必须以 Yes / No 开头, 其他题原答案要整词出现。"""
    text = str(sentence).strip()
    if "\n" in text or not 4 <= len(text.split()) <= 40:
        return False
    a, t = _norm(answer), _norm(text)
    if a in ("yes", "no"):
        return t.split()[0] == a
    return bool(a) and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", t) is not None


def apply_rewrites(samples: list[dict], rewrites: dict[int, str]) -> tuple[list[dict], dict]:
    out, applied = copy.deepcopy(samples), 0
    for i, sentence in rewrites.items():
        if not rewrite_ok(samples[i]["messages"][1]["content"], sentence):
            continue
        out[i]["messages"][0]["content"] = question_of(samples[i]) + SENTENCE_LINE
        out[i]["messages"][1]["content"] = str(sentence).strip()
        applied += 1
    return out, {"requested": len(rewrites), "applied": applied, "rejected": len(rewrites) - applied}


def description_requests(samples: list[dict]) -> list[dict]:
    seen = dict.fromkeys(s["images"][0] for s in samples if s.get("images"))
    return [{"image": p} for p in seen]


def description_samples(described: list[dict]) -> list[dict]:
    return [{"messages": [{"role": "user", "content": "<image>" + DESCRIBE},
                          {"role": "assistant", "content": d["text"].strip()}], "images": [d["image"]]}
            for d in described if str(d.get("text", "")).strip()]


def sample_extra(extra: list[dict], n: int, seed: int) -> list[dict]:
    if n > len(extra):
        raise ValueError(f"要抽 {n} 条, 文件里只有 {len(extra)} 条")
    return [extra[i] for i in sorted(random.Random(seed).sample(range(len(extra)), n))]


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(p, rows):
    Path(p).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} 条 -> {p}")


def _write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} 条 -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nofmt", "rw-requests", "rw-apply", "desc-requests", "desc-apply", "sample"])
    ap.add_argument("src", help="slake_train.json; sample 时是要抽的那个训练文件")
    ap.add_argument("out")
    ap.add_argument("--gen", help="rw-apply / desc-apply: scripts/gen_control_text.py 的输出 jsonl")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=490)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ok", type=float, default=0.95, help="rw-apply: 通过检查的比例低于它就停")
    args = ap.parse_args()
    if args.cmd == "desc-apply":                      # 这一步不读 src 的内容, 只读 --gen
        _write(args.out, description_samples(_read_jsonl(args.gen)))
        return
    rows = _read(args.src)
    if args.cmd == "nofmt":
        closed = sum(s["messages"][0]["content"].endswith(FORMAT_LINES[0]) for s in rows)
        _write(args.out, [strip_format(s) for s in rows])   # 有一条不带提示句就在这里报错停下
        print(f"全部 {len(rows)} 条都带提示句: 「是 / 否」{closed} 条, 「一个词或短语」{len(rows) - closed} 条")
    elif args.cmd == "rw-requests":
        _write_jsonl(args.out, rewrite_requests(rows, args.frac, args.seed))
    elif args.cmd == "rw-apply":
        out, stats = apply_rewrites(rows, {g["idx"]: g["text"] for g in _read_jsonl(args.gen)})
        print("改写:", stats)
        if stats["applied"] < args.min_ok * stats["requested"]:
            sys.exit(f"通过检查的不到 {args.min_ok:.0%}, 先看被拒的是哪些再决定")
        _write(args.out, out)
    elif args.cmd == "desc-requests":
        _write_jsonl(args.out, description_requests(rows))
    else:
        _write(args.out, sample_extra(rows, args.n, args.seed))


if __name__ == "__main__":
    main()
