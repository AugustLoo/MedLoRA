"""接口端到端演示: 一条命令跑完, 生成一份给人看的测试报告。

走的是和四个评估脚本完全相同的代码路径 (medvlm.model.load_model / generate + medvlm.metrics 打分),
所以这里通过, 就说明评估脚本接上这个服务也能跑。

两种模式:
  模拟模式 (没设 MEDVLM_API_BASE): 自动起一个假服务 (scripts/mock_api_server.py), 验证我们这边的代码通;
                                    假服务只会答 red / yes, 对错没有意义, 看的是链路和格式。
  真实模式 (设了 MEDVLM_API_BASE):  直接测部署好的模型, 顺带按实测速度估算四张表全跑要多久。

用法:
  python scripts/api_demo.py                                   # 模拟模式
  export MEDVLM_API_BASE=http://127.0.0.1:8000/v1
  python scripts/api_demo.py --model big35b                    # 真实模式
  python scripts/api_demo.py --model big35b --slake 10         # 再加 10 道真实 SLAKE 题 (需要 data/raw/SLAKE)

报告: outputs/api_demo/api_demo_report.md (Markdown, 可直接发给别人看) 和同名 .json
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.prompts import GENERAL_VQA, PUBMEDQA  # noqa: E402
from medvlm.remote import ENV_BASE, RemoteModel, api_base_from_env  # noqa: E402

# 四张表的请求量, 用来按实测速度估算全量评估时间
FULL_EVAL = {"SLAKE": (1061, "image"), "TextVQA": (300, "image"), "MMBench": (500, "image"),
             "PubMedQA": (1000, "text")}


def shape_image(kind: str) -> Image.Image:
    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    if kind == "red_square":
        d.rectangle([112, 112, 336, 336], fill=(220, 30, 30))
    elif kind == "blue_circle":
        d.ellipse([112, 112, 336, 336], fill=(30, 60, 220))
    elif kind == "three_dots":
        for x in (96, 224, 352):
            d.ellipse([x - 40, 184, x + 40, 264], fill=(20, 20, 20))
    return img


CTX_YES = ("A randomized trial of 400 adults found that daily aspirin reduced the rate of the primary "
           "outcome from 12% to 6% (p<0.001), with consistent effects across all subgroups.")
CTX_NO = ("A randomized trial of 400 adults found no difference in the primary outcome between vitamin C "
          "and placebo (11% vs 11%, p=0.94), and no subgroup showed any benefit.")

CASES = [
    {"id": "img-colour", "kind": "image", "image": "red_square", "max_tokens": 16, "gold": "red",
     "prompt": GENERAL_VQA.format(question="What color is the square?")},
    {"id": "img-shape", "kind": "image", "image": "blue_circle", "max_tokens": 16, "gold": "circle",
     "prompt": GENERAL_VQA.format(question="What shape is shown in the image?")},
    {"id": "img-count", "kind": "image", "image": "three_dots", "max_tokens": 16, "gold": "3",
     "prompt": GENERAL_VQA.format(question="How many black dots are there? Answer with a number.")},
    {"id": "text-yes", "kind": "text", "max_tokens": 8, "gold": "yes",
     "prompt": PUBMEDQA.format(context=CTX_YES, question="Does daily aspirin reduce the primary outcome?")},
    {"id": "text-no", "kind": "text", "max_tokens": 8, "gold": "no",
     "prompt": PUBMEDQA.format(context=CTX_NO, question="Does vitamin C reduce the primary outcome?")},
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock():
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "mock_api_server.py"), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/models", timeout=1)
            return proc, base
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    sys.exit("模拟服务没起来")


def score(pred: str, gold: str) -> bool:
    p = M.normalize(pred)
    return p == M.normalize(gold) or M.normalize(gold) in p.split()


def truncate_payload(payload: dict) -> dict:
    """报告里展示请求体时, 把几十 KB 的 base64 图片截短。"""
    p = json.loads(json.dumps(payload))
    for m in p.get("messages", []):
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if part.get("type") == "image_url":
                    u = part["image_url"]["url"]
                    part["image_url"]["url"] = u[:48] + f"...(共 {len(u)} 字符)"
    return p


def run_slake(model, processor, n: int) -> list[dict]:
    from medvlm.prompts import slake_prompt
    from medvlm.slake import load_split, open_image
    rows = load_split("test", lang="en")[:n]
    out = []
    for r in rows:
        t = time.time()
        pred = generate(model, processor, slake_prompt(r["question"], r["answer_type"]), open_image(r), 32)
        dt = time.time() - t
        closed = str(r["answer_type"]).upper() == "CLOSED"
        ok = M.closed_score(pred, r["answer"]) if closed else M.exact_match(pred, r["answer"])
        out.append({"id": f"slake-{r['qid']}", "kind": "image", "question": r["question"], "gold": r["answer"],
                    "pred": pred, "pass": bool(ok), "seconds": round(dt, 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="服务端模型名; 不填就用 /v1/models 返回的第一个")
    ap.add_argument("--slake", type=int, default=0, help="额外跑 N 道真实 SLAKE 测试题 (需要 data/raw/SLAKE)")
    args = ap.parse_args()

    mock_proc = None
    if not api_base_from_env():
        mock_proc, base = start_mock()
        os.environ[ENV_BASE] = base
        mode = "模拟模式 (假服务, 只验证链路和格式, 对错无意义)"
    else:
        mode = "真实模式"
    try:
        probe = RemoteModel.from_env(args.model or "", max_pixels=512 * 28 * 28)
        served = probe.list_models()
        name = args.model or (served[0] if served else "")
        if not name:
            sys.exit("服务端没有返回模型名, 请用 --model 指定")

        # 与评估脚本同一入口
        model, processor = load_model(name, None, max_pixels=512 * 28 * 28)
        sample_payload = truncate_payload(model.build_payload(CASES[0]["prompt"], shape_image("red_square"), 16))

        results = []
        for c in CASES:
            img = shape_image(c["image"]) if c["kind"] == "image" else None
            t = time.time()
            try:
                pred, err = generate(model, processor, c["prompt"], img, c["max_tokens"]), ""
            except (RuntimeError, SystemExit) as e:
                pred, err = "", str(e)
            dt = time.time() - t
            results.append({"id": c["id"], "kind": c["kind"], "gold": c["gold"], "pred": pred,
                            "pass": (not err) and score(pred, c["gold"]), "error": err[:200],
                            "seconds": round(dt, 2)})
            print(f"  {c['id']:<11} gold={c['gold']:<7} pred={pred!r:<20} {'OK' if results[-1]['pass'] else '--'}"
                  f"  {dt:.2f}s {err[:80]}")
        if args.slake:
            print(f"  + {args.slake} 道 SLAKE 真题")
            results += run_slake(model, processor, args.slake)
    finally:
        if mock_proc:
            mock_proc.terminate()

    lat = {k: [r["seconds"] for r in results if r["kind"] == k and not r.get("error")] for k in ("image", "text")}
    avg = {k: (sum(v) / len(v) if v else None) for k, v in lat.items()}
    est = None
    if avg["image"] is not None and avg["text"] is not None:
        est = sum(n * avg[kind] for n, kind in FULL_EVAL.values())
    errors = [r for r in results if r.get("error")]
    long_answers = [r for r in results if len(r["pred"].split()) > 4]
    link_ok = not errors and all(r["pred"] for r in results)

    info = backend_info(model)
    report = {
        "time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "host": platform.node(),
        "backend": info, "served_models": served, "results": results,
        "avg_seconds": avg, "estimated_full_eval_seconds": est,
        "link_ok": link_ok, "n_pass": sum(r["pass"] for r in results), "n": len(results),
    }
    out_dir = REPO / "outputs" / "api_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "api_demo_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("# 接口端到端测试报告\n")
    L.append(f"- 时间：{report['time']}，机器：`{report['host']}`")
    L.append(f"- 模式：**{mode}**")
    L.append(f"- 服务地址：`{info['api_base']}`，模型名：`{info['served_model']}`，服务端列出的模型：`{served}`")
    L.append(f"- 解码：temperature 0，top_p 1；图片先等比缩小到不超过 {info['max_pixels']} 像素，PNG 无损编码")
    if info.get("extra"):
        L.append(f"- 附加请求参数：`{json.dumps(info['extra'], ensure_ascii=False)}`")
    L.append("")
    L.append(f"## 结论：链路{'通' if link_ok else '不通'}，答对 {report['n_pass']} / {report['n']}\n")
    if "模拟" in mode:
        L.append("模拟模式下假服务只会答 red / yes，所以答对数没有意义。这一步证明的是："
                 "请求能发出去、格式被服务端接受、回答能拿回来、打分代码能跑。\n")
    if errors:
        L.append(f"有 {len(errors)} 题请求失败，见下表「错误」一列。\n")
    if long_answers:
        L.append(f"有 {len(long_answers)} 题回答超过 4 个词。评估要求只答一个词或一个短语，"
                 "回答偏长会被判错。检查模型是否在输出思考段或解释，见 docs/API.md 第四节。\n")
    L.append("## 逐题结果\n")
    L.append("| 题目 | 类型 | 标准答案 | 模型回答 | 对错 | 耗时 (秒) | 错误 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['id']} | {'看图' if r['kind'] == 'image' else '文字'} | {r['gold']} | "
                 f"{r['pred'] or '（空）'} | {'✓' if r['pass'] else '✗'} | {r['seconds']} | {r.get('error', '')} |")
    L.append("")
    L.append("## 速度\n")
    L.append(f"- 看图题平均 {avg['image']:.2f} 秒，文字题平均 {avg['text']:.2f} 秒" if avg["image"] is not None and avg["text"] is not None
             else "- 速度数据不全（有一类题全部失败）")
    if est is not None and "模拟" not in mode:
        L.append(f"- 按这个速度，四张表共 2,861 题，逐题顺序跑约 **{est / 3600:.1f} 小时**")
    L.append("")
    L.append("## 实际发出的请求（第一题，图片已截短）\n")
    L.append("```json")
    L.append(json.dumps(sample_payload, indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")
    L.append("生成方式：`python scripts/api_demo.py`。评估脚本调用模型走的是同一个入口 "
             "(`medvlm.model.load_model` / `generate`)，这里通过，四张表的评估就能直接接上。")
    md = "\n".join(L) + "\n"
    (out_dir / "api_demo_report.md").write_text(md, encoding="utf-8")
    print(f"\n链路{'通' if link_ok else '不通'}，答对 {report['n_pass']} / {report['n']}")
    print(f"报告: {out_dir / 'api_demo_report.md'}")


if __name__ == "__main__":
    main()
