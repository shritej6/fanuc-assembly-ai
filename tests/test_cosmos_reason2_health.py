import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from serving.cosmos_reason2 import health


def _server(model_id: str, root: str, content: str) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return

        def _send(self, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send({"data": [{"id": model_id, "root": root}]})

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            message = {"content": "<think>checked the top part</think>" + content}
            self._send({"model": model_id, "choices": [{"message": message, "finish_reason": "stop"}]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize(
    ("model_id", "root", "content", "code"),
    [
        ("Cosmos-Reason2-8B", "nvidia/Cosmos-Reason2-8B", '{"verdict": "pass", "reason": "seated"}', 0),
        ("Cosmos-Reason2-8B", "nvidia/Cosmos-Reason2-8B", "the part looks fine", 1),
        ("Cosmos-Reason2-2B", "nvidia/Cosmos-Reason2-2B", '{"verdict": "pass", "reason": "seated"}', 1),
        ("Cosmos-Reason2-8B", "/models/some-finetune", '{"verdict": "pass", "reason": "seated"}', 1),
    ],
)
def test_health_requires_official_8b_and_parseable_answers_over_http(
    monkeypatch: pytest.MonkeyPatch, model_id: str, root: str, content: str, code: int
) -> None:
    server = _server(model_id, root, content)
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        monkeypatch.setattr(sys, "argv", ["health", "--base-url", url])
        assert health.main() == code
    finally:
        server.shutdown()
