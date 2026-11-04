"""Tests for the scribe chat client, against a local HTTP server and the fake."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from scribe.client import ChatClient, ClientError, FakeClient, user_message

SECRET = "sk-test-not-a-real-key"


class _Handler(BaseHTTPRequestHandler):
    plan: list[int] = []
    seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append({"auth": self.headers.get("Authorization"), "body": body, "path": self.path})
        code = _Handler.plan.pop(0) if _Handler.plan else 200
        if code != 200:
            self.send_response(code)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error": "busy"}')
            return
        out = {"model": body["model"], "choices": [{"message": {"role": "assistant", "content": "B"}}],
               "usage": {"prompt_tokens": 11, "completion_tokens": 1}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.plan, _Handler.seen = [], []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _client(base: str, ledger: Path) -> ChatClient:
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return ChatClient(api_base=base, api_key=SECRET, ledger=ledger, backoff=0.0,
                      max_retries=2, sleep=lambda s: None, opener=direct)


def test_chat_retries_on_429_and_5xx_and_records_usage(server: str, tmp_path: Path) -> None:
    _Handler.plan = [429, 503]
    led = tmp_path / "usage.jsonl"
    client = _client(server, led)
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    reply = client.chat("vision-x", [user_message("which?", images=[img])], temperature=0.0, max_tokens=5)
    assert reply.text == "B"
    assert reply.usage.attempts == 3 and reply.usage.prompt_tokens == 11
    assert len(_Handler.seen) == 3 and _Handler.seen[0]["path"] == "/v1/chat/completions"
    part = _Handler.seen[0]["body"]["messages"][0]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    assert _Handler.seen[0]["body"]["max_tokens"] == 5
    rec = [json.loads(x) for x in led.read_text().splitlines()]
    assert rec[0]["model"] == "vision-x" and rec[0]["status"] == "ok"
    assert SECRET not in led.read_text() and SECRET not in repr(client)


def test_chat_gives_up_and_raises(server: str, tmp_path: Path) -> None:
    _Handler.plan = [500, 500, 500]
    client = _client(server, tmp_path / "u.jsonl")
    with pytest.raises(ClientError) as info:
        client.chat("m", [user_message("hi")])
    assert SECRET not in str(info.value)
    assert json.loads((tmp_path / "u.jsonl").read_text())["status"] == "error"


def test_chat_does_not_retry_on_401(server: str, tmp_path: Path) -> None:
    _Handler.plan = [401, 200]
    with pytest.raises(ClientError, match="401"):
        _client(server, tmp_path / "u.jsonl").chat("m", [user_message("hi")])
    assert len(_Handler.seen) == 1


def test_missing_base_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("SCRIBE_API_BASE", raising=False)
    with pytest.raises(ValueError, match="SCRIBE_API_BASE"):
        ChatClient()


def test_fake_client_scripts_and_raises(tmp_path: Path) -> None:
    fake = FakeClient(["one", ClientError("down"), "two"], ledger=tmp_path / "f.jsonl")
    assert fake.chat("a", [user_message("x")]).text == "one"
    with pytest.raises(ClientError):
        fake.chat("a", [user_message("x")])
    assert fake.chat("b", [user_message("y")], tag="t").text == "two"
    assert [c[0] for c in fake.calls] == ["a", "a", "b"]
    assert len((tmp_path / "f.jsonl").read_text().splitlines()) == 3


def test_failed_call_keeps_its_error_text_in_the_ledger(server: str, tmp_path: Path) -> None:
    _Handler.plan = [401]
    with pytest.raises(ClientError):
        _client(server, tmp_path / "u.jsonl").chat("m", [user_message("hi")])
    import json

    rec = json.loads((tmp_path / "u.jsonl").read_text().splitlines()[-1])
    assert rec["status"] == "error" and "401" in rec["error"]
