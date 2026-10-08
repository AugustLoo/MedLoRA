"""对比提问小工具: 向对比服务 (train/interns2/compare_serve.sh) 当前开着的模型提问, 答案按模型分别存档, 最后按题合并。

对比服务一次只开一个 35B, 所以流程是: 开原版 → batch 跑一遍 → 切到训练版 → 同一个问题文件再跑一遍 → merge。
请求设置与我们的评测一致: 关闭思考模式、贪心解码 (temperature 0)、图片先等比缩到不超过 512×28×28 像素。
说明见 docs/COMPARE_API.md。

用法 (服务器上, 有 Pillow 的 Python 环境即可, 例如 conda activate /home/ubuntu/chunqian/envs/s2train):
  python scripts/compare_ask.py ask "What modality is this image?" --image some.png
  python scripts/compare_ask.py batch questions.jsonl        # → outputs/compare/answers_<模型名>.jsonl
  python scripts/compare_ask.py merge                         # → outputs/compare/compare.jsonl

问题文件 (JSONL) 每行一题: {"id": "q1", "question": "...", "image": "可选: 图片路径"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.remote import RemoteModel  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:23334/v1"
OUT_DIR = REPO / "outputs" / "compare"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_PIXELS = 512 * 28 * 28  # 与评测时的远程客户端相同
NAMES = ("intern-s2-base", "intern-s2-medlora")
# 服务按评测时的参数启动, /v1/models 返回的是模型目录路径; 按目录名换成好认的名字
DIR_NAMES = {"Intern-S2-Preview": "intern-s2-base", "Intern-S2-Preview-MedLoRA": "intern-s2-medlora"}


def friendly(served_id: str) -> str:
    leaf = served_id.rstrip("/").replace("\\", "/").rsplit("/", 1)[-1]
    return DIR_NAMES.get(leaf, leaf)


def connect(base: str, max_tokens: int) -> RemoteModel:
    probe = RemoteModel(base, model="", retries=1, timeout=10)
    try:
        served = probe.list_models()
    except RuntimeError as e:
        sys.exit(f"连不上 {base}: {e}\n对比服务开着吗? 切换中要等几分钟; 用 bash train/interns2/compare_serve.sh status 查看。")
    if not served:
        sys.exit(f"{base} 没有返回模型名")
    return RemoteModel(base, model=served[0], max_pixels=MAX_PIXELS, timeout=600, extra=NO_THINK, min_tokens=max_tokens)


def answer(model: RemoteModel, question: str, image: str | None, max_tokens: int) -> dict:
    img = None
    if image:
        if not Path(image).is_file():
            sys.exit(f"图片找不到: {image}")
        from PIL import Image
        img = Image.open(image)
    t0 = time.time()
    text = model.chat(question, img, max_tokens)
    return {"model": friendly(model.model), "served_id": model.model, "answer": text,
            "finish_reason": model.last.get("finish_reason"), "seconds": round(time.time() - t0, 1)}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path} 第 {n} 行不是合法 JSON: {e}")
    return rows


def cmd_ask(args):
    model = connect(args.base, args.max_tokens)
    r = answer(model, args.question, args.image, args.max_tokens)
    if args.json:
        print(json.dumps({"question": args.question, "image": args.image, **r}, ensure_ascii=False))
    else:
        cut = "  (达到长度上限, 被截断)" if r["finish_reason"] == "length" else ""
        print(f"[{r['model']}  {r['seconds']}s]{cut}\n{r['answer']}")


def cmd_batch(args):
    qs = read_jsonl(Path(args.questions))
    ids = [q.get("id") for q in qs]
    if any(i in (None, "") for i in ids) or len(set(ids)) != len(ids):
        sys.exit("问题文件里每题都要有不重复的 id")
    for q in qs:
        if not q.get("question"):
            sys.exit(f"题 {q['id']} 没有 question")
        if q.get("image") and not Path(q["image"]).is_file():
            sys.exit(f"题 {q['id']} 的图片找不到: {q['image']}")
    model = connect(args.base, args.max_tokens)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = friendly(model.model)
    out = Path(args.out) if args.out else OUT_DIR / f"answers_{name}.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.is_file() else set()
    todo = [q for q in qs if q["id"] not in done]
    print(f"模型 {name} | 共 {len(qs)} 题, 已答 {len(qs) - len(todo)}, 本次 {len(todo)} → {out}")
    with out.open("a", encoding="utf-8") as f:
        for k, q in enumerate(todo, 1):
            r = answer(model, q["question"], q.get("image"), args.max_tokens)
            f.write(json.dumps({"id": q["id"], "question": q["question"], "image": q.get("image"), **r},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {k}/{len(todo)}  {q['id']}  {r['seconds']}s", flush=True)
    print("完成。另一个模型也跑完后: python scripts/compare_ask.py merge")


def cmd_merge(args):
    paths = [Path(p) for p in args.files] if args.files else [OUT_DIR / f"answers_{n}.jsonl" for n in NAMES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit("找不到答案文件: " + ", ".join(missing) + "\n两个模型都用 batch 跑过同一个问题文件了吗?")
    tables = []
    for p in paths:
        rows = read_jsonl(p)
        if not rows:
            sys.exit(f"{p} 是空的")
        tables.append((rows[0]["model"], {r["id"]: r for r in rows}))
    order = list(tables[0][1])
    for _, t in tables[1:]:
        order += [i for i in t if i not in order]
    out = Path(args.out) if args.out else OUT_DIR / "compare.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    gaps = 0
    with out.open("w", encoding="utf-8") as f:
        for i in order:
            first = next(t[i] for _, t in tables if i in t)
            row = {"id": i, "question": first["question"], "image": first.get("image")}
            for name, t in tables:
                row[name] = t[i]["answer"] if i in t else None
                gaps += i not in t
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    names = " vs ".join(n for n, _ in tables)
    print(f"{names}: {len(order)} 题 → {out}" + (f"  (有 {gaps} 处缺答案, 记为 null)" if gaps else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"接口地址, 默认 {DEFAULT_BASE}")
    ap.add_argument("--max-tokens", type=int, default=512, help="最长回答 token 数, 默认 512")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="问一道题")
    a.add_argument("question")
    a.add_argument("--image")
    a.add_argument("--json", action="store_true", help="输出一行 JSON (给程序读)")
    b = sub.add_parser("batch", help="跑一个问题文件")
    b.add_argument("questions")
    b.add_argument("--out", help="默认 outputs/compare/answers_<模型名>.jsonl")
    m = sub.add_parser("merge", help="按题合并两个模型的答案")
    m.add_argument("files", nargs="*", help="默认原版与训练版的两个答案文件")
    m.add_argument("--out", help="默认 outputs/compare/compare.jsonl")
    args = ap.parse_args()
    {"ask": cmd_ask, "batch": cmd_batch, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
