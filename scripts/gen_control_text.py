"""让正在服务的原版模型生成对照实验要用的文字 (R2 的改写句, R1 的图片描述)。

请求文件由 data/make_short_answer_controls.py 的 rw-requests / desc-requests 生成。
中断了重跑会跳过已经生成的条目。只走本机的模型服务 (MEDVLM_API_BASE), 不调外部服务。

用法 (主机上, 先 bash train/interns2/serve.sh start <原版模型目录>):
  export MEDVLM_API_BASE=http://127.0.0.1:23334/v1
  export MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  python scripts/gen_control_text.py rewrite  rw_req.jsonl   rw_gen.jsonl   --model <原版模型目录>
  python scripts/gen_control_text.py describe desc_req.jsonl desc_gen.jsonl --model <原版模型目录> \
      --path-map /workspace/chunqian/MedLoRA/data/raw/SLAKE=/home/ubuntu/chunqian/data/SLAKE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data"))

KEY = {"rewrite": "idx", "describe": "image"}
MAX_NEW_TOKENS = {"rewrite": 64, "describe": 160}


def map_path(p: str, path_map: list[tuple[str, str]]) -> str:
    for old, new in path_map:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def todo(requests: list[dict], done: list[dict], key: str) -> list[dict]:
    have = {d[key] for d in done}
    return [r for r in requests if r[key] not in have]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rewrite", "describe"])
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, help="服务里的模型名 (/v1/models 返回的那个, 即原版模型目录)")
    ap.add_argument("--path-map", action="append", default=[], help="旧前缀=新前缀, 把训练文件里的图片路径换成本机路径")
    ap.add_argument("--concurrency", type=int, default=8, help="同时发几个请求 (与 eval_chexpert.py 相同)")
    ap.add_argument("--limit", type=int, default=0, help="只生成前几条, 试跑用")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from make_short_answer_controls import DESCRIBE
    from medvlm.model import generate, load_model

    key, out = KEY[args.mode], Path(args.out)
    path_map = [tuple(x.split("=", 1)) for x in args.path_map]
    rest = todo(read_jsonl(Path(args.requests)), read_jsonl(out), key)
    if args.limit:
        rest = rest[: args.limit]
    print(f"还要生成 {len(rest)} 条 ({args.mode})", flush=True)
    model, processor = load_model(args.model)

    def ask(r):
        if args.mode == "rewrite":
            return generate(model, processor, r["prompt"], None, MAX_NEW_TOKENS[args.mode])
        image = Image.open(map_path(r["image"], path_map)).convert("RGB")
        return generate(model, processor, DESCRIBE, image, MAX_NEW_TOKENS[args.mode])

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, (r, text) in enumerate(zip(rest, ex.map(ask, rest)), 1):   # map 按原顺序返回, 文件里的顺序不变
            f.write(json.dumps({key: r[key], "text": text.strip()}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(rest):
                print(f"{i}/{len(rest)}", flush=True)


if __name__ == "__main__":
    main()
