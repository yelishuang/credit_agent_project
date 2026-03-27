#!/usr/bin/env python3
"""
专家评测 HTTP 服务 — 提供前端页面和 API 端点。

用法：
  python -m src.evaluation.serve_expert_eval --port 8080 --eval-dir outputs/evaluation/20260323_150000
"""

import argparse
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_CASES_PATH

HTML_PATH = Path(__file__).resolve().parent / "static" / "expert_eval.html"


class ExpertEvalHandler(BaseHTTPRequestHandler):
    eval_dir: Path = None
    results_lock = threading.Lock()

    def log_message(self, format, *args):
        print(f"[ExpertEval] {args[0]}")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            html = HTML_PATH.read_bytes()
            self._send_html(html)
        elif path == "/api/data":
            self._handle_get_data()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/result":
            self._handle_post_result()
        else:
            self.send_error(404)

    def _handle_get_data(self):
        responses_path = self.eval_dir / "responses.json"
        test_cases_path = TEST_CASES_PATH
        if not responses_path.exists():
            self._send_json({"error": "responses.json not found"}, 404)
            return
        with open(responses_path, "r", encoding="utf-8") as f:
            responses = json.load(f)
        with open(test_cases_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        self._send_json({"responses": responses, "test_cases": test_cases})

    def _handle_post_result(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        results_path = self.eval_dir / "expert_eval_results.json"

        with self.results_lock:
            existing = []
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(body)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

        self._send_json({"status": "ok", "total": len(existing)})


def main():
    parser = argparse.ArgumentParser(description="专家评测 HTTP 服务")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--eval-dir", required=True, help="评测目录路径")
    args = parser.parse_args()

    ExpertEvalHandler.eval_dir = Path(args.eval_dir)
    server = HTTPServer(("0.0.0.0", args.port), ExpertEvalHandler)
    print(f"Expert eval server running at http://localhost:{args.port}")
    print(f"Eval dir: {args.eval_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
