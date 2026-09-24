"""远程推理后端: 通过 OpenAI 兼容的 HTTP 接口调用部署在别处的模型 (例如同学容器里的 35B)。

为什么是 OpenAI 兼容接口: vLLM / SGLang / LMDeploy / Ollama 都原生提供
`POST /v1/chat/completions`, 服务端用哪个框架都能对上, 我们这边不用改。

用法 (四个评估脚本和 train/eval_all.sh 都不用改, 只设环境变量):
    export MEDVLM_API_BASE=http://127.0.0.1:8000/v1      # 服务地址, 设了就走远程模式
    export MEDVLM_API_KEY=...                              # 服务端要求鉴权时才设, 不要写进代码或 git
    MODEL=<服务端的模型名> bash train/eval_all.sh big35b_base

可选:
    MEDVLM_API_TIMEOUT   单次请求超时秒数, 默认 180
    MEDVLM_API_EXTRA     JSON, 原样并入请求体。例如关闭思考模式:
                         '{"chat_template_kwargs": {"enable_thinking": false}}'

与本地推理保持一致的三件事:
    1. 贪心解码: temperature=0, top_p=1 (本地是 do_sample=False)
    2. 最大生成长度: 与各评估脚本传入的 max_new_tokens 相同
    3. 图片像素上限: 客户端先按 max_pixels 等比缩小再发, 与本地 processor 的 max_pixels 同一口径
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

from PIL import Image

ENV_BASE = "MEDVLM_API_BASE"
ENV_KEY = "MEDVLM_API_KEY"
ENV_TIMEOUT = "MEDVLM_API_TIMEOUT"
ENV_EXTRA = "MEDVLM_API_EXTRA"

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def api_base_from_env() -> str | None:
    base = os.environ.get(ENV_BASE, "").strip().rstrip("/")
    return base or None


def encode_image(image: Image.Image, max_pixels: int) -> str:
    """等比缩小到不超过 max_pixels, PNG 无损编码成 data URL (医学影像不用有损压缩)。"""
    img = image.convert("RGB")
    w, h = img.size
    if max_pixels and w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clean_output(text: str | None) -> str:
    """去掉推理模型的 <think>…</think> 段; 只剩未闭合的 <think> 时说明被截断, 返回空串 (记为答错)。"""
    t = _THINK.sub("", text or "")
    if "<think>" in t.lower():
        return ""
    return t.strip()


class RemoteModel:
    def __init__(self, base: str, model: str, api_key: str | None = None, max_pixels: int = 512 * 28 * 28,
                 timeout: float = 180, retries: int = 4, extra: dict | None = None):
        self.base = base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_pixels = max_pixels
        self.timeout = timeout
        self.retries = retries
        self.extra = extra or {}

    @classmethod
    def from_env(cls, model: str, max_pixels: int) -> "RemoteModel":
        base = api_base_from_env()
        if not base:
            raise RuntimeError(f"没有设置 {ENV_BASE}")
        extra = os.environ.get(ENV_EXTRA, "").strip()
        return cls(base, model, api_key=os.environ.get(ENV_KEY) or None, max_pixels=max_pixels,
                   timeout=float(os.environ.get(ENV_TIMEOUT, "180")),
                   extra=json.loads(extra) if extra else None)

    # ---------- HTTP ----------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    last = f"HTTP {e.code}: {body}"
                else:
                    raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = repr(e)
                if attempt >= self.retries:
                    raise RuntimeError(f"{method} {url} 连不上 ({last})。服务起来了吗? 地址和端口对吗?") from None
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"{method} {url} 重试 {self.retries} 次仍失败: {last}")

    def list_models(self) -> list[str]:
        return [m.get("id", "") for m in self._request("GET", "/models").get("data", [])]

    def chat(self, prompt: str, image: Image.Image | None = None, max_new_tokens: int = 32) -> str:
        if image is not None:
            content = [{"type": "image_url", "image_url": {"url": encode_image(image, self.max_pixels)}},
                       {"type": "text", "text": prompt}]
        else:
            content = prompt
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_new_tokens, "temperature": 0, "top_p": 1}
        payload.update(self.extra)
        out = self._request("POST", "/chat/completions", payload)
        msg = (out.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content")
        if isinstance(text, list):  # 少数服务端把 content 返回成分段列表
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return clean_output(text)

    def describe(self) -> dict:
        return {"backend": "api", "api_base": self.base, "served_model": self.model,
                "max_pixels": self.max_pixels, "extra": self.extra}
