"""接口连通性检查: 正式评估前先跑这个, 一分钟内确认四件事。

  1. 服务连得上, 列出服务端的模型名 (评估时 MODEL 要填其中一个)
  2. 纯文字题能答 (PubMedQA 用)
  3. 看图题能答 (SLAKE / TextVQA / MMBench 用); 纯文字模型会在这一步报错
  4. 回答干不干净: 按「只回答一个词」的要求, 有没有多余的解释或思考段

用法:
  export MEDVLM_API_BASE=http://127.0.0.1:8000/v1
  python scripts/check_api.py                      # 自动用服务端第一个模型名
  python scripts/check_api.py --model <模型名>
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.remote import RemoteModel, api_base_from_env  # noqa: E402


def test_image() -> Image.Image:
    """一张不需要任何数据集的测试图: 白底上一个红色方块。"""
    img = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(img).rectangle([112, 112, 336, 336], fill=(220, 30, 30))
    return img


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if not api_base_from_env():
        sys.exit("先设置 MEDVLM_API_BASE, 例如 export MEDVLM_API_BASE=http://127.0.0.1:8000/v1")

    probe = RemoteModel.from_env(args.model or "", max_pixels=512 * 28 * 28)
    print(f"[1] 服务地址 {probe.base}")
    models = probe.list_models()
    print(f"    服务端模型: {models}")
    if not models and not args.model:
        sys.exit("    服务端没有返回模型名, 请用 --model 指定")
    probe.model = args.model or models[0]
    print(f"    本次使用: {probe.model}")

    ans, dt = timed(lambda: probe.chat(
        "Is the sky blue on a clear day? Answer with exactly one word: yes, no, or maybe.", None, 8))
    print(f"[2] 纯文字题: {ans!r}  ({dt:.1f}s)")

    try:
        ans, dt = timed(lambda: probe.chat(
            "What color is the square in the image?\nAnswer with a single word or short phrase.",
            test_image(), 16))
        print(f"[3] 看图题: {ans!r}  ({dt:.1f}s)")
        img_ok = "red" in ans.lower()
    except RuntimeError as e:
        print(f"[3] 看图题失败: {e}")
        print("    多半是纯文字模型, 或服务端没开多模态。SLAKE / TextVQA / MMBench 三张表做不了。")
        img_ok = False

    words = len(ans.split())
    print(f"[4] 回答长度 {words} 个词" + ("  <- 偏长, 检查是否有思考段或解释, 见 docs/API.md" if words > 4 else ""))
    print("\n结论:", "可以开始评估" if img_ok else "看图题没通过, 先和服务端对一下")


if __name__ == "__main__":
    main()
