"""A small OpenAI-compatible chat client over plain HTTPS, standard library only.

Configuration comes from the environment:

``SCRIBE_API_BASE``
    The API root, for example ``https://openrouter.ai/api/v1`` or
    ``https://router.huggingface.co/v1``. ``/chat/completions`` is appended.
``SCRIBE_API_KEY``
    The bearer token. It is sent in the Authorization header and nowhere
    else: never printed, logged or written to the ledger.
``SCRIBE_CA_BUNDLE``
    Optional CA bundle path. Otherwise ``SSL_CERT_FILE`` is used if set,
    then the system default. ``HTTPS_PROXY`` is honoured by urllib.

Every call appends one usage line to the JSONL ledger, if one is given:
model, prompt and completion tokens, latency, attempts, status and a tag.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

RETRY_STATUS = {429, 500, 502, 503, 504}


class ClientError(RuntimeError):
    """A call failed after its retries. Callers treat this as an infrastructure failure."""


@dataclass
class Usage:
    """One call's cost record, as written to the ledger."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    status: str = "ok"
    tag: str | None = None
    #: why the model stopped, as the API reports it (``stop``, ``length``...); None if not given
    finish_reason: str | None = None
    #: the router's own cost for the call in dollars, when it reports one; None otherwise
    cost: float | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Reply:
    """The assistant's text and the usage record of the call that produced it."""

    text: str
    usage: Usage


def png_data_url(image: np.ndarray | bytes | str | Path) -> str:
    """A base64 PNG data URL from a uint8 array, PNG bytes or an image file path."""
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
    elif isinstance(image, bytes):
        data = image
    else:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG")
        data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def user_message(text: str, images: list[Any] | None = None) -> dict[str, Any]:
    """A user message with text and optional images, in the OpenAI content-part format."""
    if not images:
        return {"role": "user", "content": text}
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(img)}})
    return {"role": "user", "content": parts}


class _Ledger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, usage: Usage) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as fh:
            fh.write(json.dumps(asdict(usage)) + "\n")


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SCRIBE_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if cafile and Path(cafile).is_file():
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


class ChatClient:
    """Chat completions against an OpenAI-compatible endpoint.

    ``chat(model, messages, ...)`` returns a ``Reply`` or raises
    ``ClientError`` once retries on 429, 5xx and connection errors are used
    up. Other HTTP errors (400, 401, 404) raise at once.
    """

    def __init__(self, api_base: str | None = None, api_key: str | None = None,
                 ledger: str | Path | None = None, timeout: float = 180.0, max_retries: int = 5,
                 backoff: float = 2.0, sleep: Callable[[float], None] = time.sleep,
                 opener: urllib.request.OpenerDirector | None = None) -> None:
        base = api_base or os.environ.get("SCRIBE_API_BASE", "")
        if not base:
            raise ValueError("set SCRIBE_API_BASE, for example https://openrouter.ai/api/v1")
        self.url = base.rstrip("/") + "/chat/completions"
        self._key = api_key if api_key is not None else os.environ.get("SCRIBE_API_KEY", "")
        self.ledger = _Ledger(ledger)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=_ssl_context()))

    def __repr__(self) -> str:  # never show the key
        return f"ChatClient(url={self.url!r}, key={'set' if self._key else 'unset'})"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        delay = self.backoff * (2 ** attempt) * (0.75 + 0.5 * random.random())
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        self._sleep(min(delay, 120.0))

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.8,
             max_tokens: int = 4096, tag: str | None = None, **extra: Any) -> Reply:
        """Send one chat completion request and return the first choice's text."""
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **extra}
        data = json.dumps(body).encode("utf-8")
        start = time.monotonic()
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=self._headers(), method="POST")
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return self._finish(model, payload, start, attempt + 1, tag)
            except urllib.error.HTTPError as exc:
                snippet = exc.read()[:200].decode("utf-8", "replace") if exc.fp else ""
                last = f"HTTP {exc.code}: {snippet}"
                if exc.code not in RETRY_STATUS or attempt == self.max_retries:
                    break
                self._wait(attempt, exc.headers.get("Retry-After") if exc.headers else None)
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
                if attempt == self.max_retries:
                    break
                self._wait(attempt, None)
        self.ledger.write(Usage(model=model, latency_s=round(time.monotonic() - start, 3),
                                attempts=attempt + 1, status="error", tag=tag))
        raise ClientError(f"{model}: {last}")

    def _finish(self, model: str, payload: dict[str, Any], start: float, attempts: int,
                tag: str | None) -> Reply:
        try:
            msg = payload["choices"][0]["message"]
            text = msg.get("content") or ""
            if isinstance(text, list):  # some routers return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            self.ledger.write(Usage(model=model, latency_s=round(time.monotonic() - start, 3),
                                    attempts=attempts, status="error", tag=tag))
            raise ClientError(f"{model}: response had no choices[0].message") from exc
        u = payload.get("usage") or {}
        usage = Usage(model=payload.get("model", model), prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                      completion_tokens=int(u.get("completion_tokens", 0) or 0),
                      latency_s=round(time.monotonic() - start, 3), attempts=attempts, tag=tag,
                      finish_reason=payload["choices"][0].get("finish_reason"),
                      cost=float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else None)
        self.ledger.write(usage)
        return Reply(text=text, usage=usage)


class FakeClient:
    """A scripted stand-in for ``ChatClient`` in tests.

    ``script`` is either a list of replies used in order or a function
    ``(model, messages) -> str``. A reply that is an exception instance is
    raised instead of returned, to simulate infrastructure failures. Every
    call is recorded in ``calls`` as ``(model, messages, kwargs)``.
    """

    def __init__(self, script: list[Any] | Callable[[str, list[dict[str, Any]]], Any],
                 ledger: str | Path | None = None) -> None:
        self.script = script
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        self.ledger = _Ledger(ledger)
        self._i = 0

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.8,
             max_tokens: int = 4096, tag: str | None = None, **extra: Any) -> Reply:
        """Return the next scripted reply, recording the call."""
        self.calls.append((model, [dict(m) for m in messages],
                           {"temperature": temperature, "max_tokens": max_tokens, "tag": tag, **extra}))
        if callable(self.script):
            out = self.script(model, messages)
        else:
            if self._i >= len(self.script):
                raise ClientError("FakeClient script ran out")
            out = self.script[self._i]
            self._i += 1
        if isinstance(out, BaseException):
            self.ledger.write(Usage(model=model, status="error", tag=tag))
            raise out
        usage = Usage(model=model, prompt_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                      completion_tokens=len(str(out)) // 4, tag=tag)
        self.ledger.write(usage)
        return Reply(text=str(out), usage=usage)
