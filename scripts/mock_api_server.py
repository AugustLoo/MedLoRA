"""假的 OpenAI 兼容服务, 只用来测试我们这边的调用代码, 不需要任何模型或显卡。

规则: 请求里带图片就回答 "red", 否则回答 "yes"。同时检查请求格式,
不符合约定 (没有 model 字段、temperature 不是 0、图片不是 data URL) 就返回 HTTP 400。

用法:
  python scripts/mock_api_server.py --port 8765
  # 另一个终端
  export MEDVLM_API_BASE=http://127.0.0.1:8765/v1
  python scripts/check_api.py
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "mock-vlm"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        problems = []
        if not req.get("model"):
            problems.append("missing model")
        if req.get("temperature") != 0:
            problems.append("temperature must be 0")
        has_image = False
        for m in req.get("messages", []):
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        has_image = True
                        if not part["image_url"]["url"].startswith("data:image/"):
                            problems.append("image is not a data URL")
        if problems:
            return self._send(400, {"error": "; ".join(problems)})
        answer = "red" if has_image else "yes"
        self._send(200, {"id": "mock", "object": "chat.completion", "model": req["model"],
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": answer}}]})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-name", default=MODEL_NAME, help="/v1/models 返回的模型名 (测 compare_ask.py 时用)")
    args = ap.parse_args()
    globals()["MODEL_NAME"] = args.model_name
    print(f"mock server on http://127.0.0.1:{args.port}/v1  (model: {MODEL_NAME})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
