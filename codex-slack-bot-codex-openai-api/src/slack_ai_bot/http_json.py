from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class JsonApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            **(headers or {}),
        },
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = body
        raise JsonApiError(f"POST {url} failed with HTTP {exc.code}", exc.code, payload) from exc


def get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    req = request.Request(url, method="GET", headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = body
        raise JsonApiError(f"GET {url} failed with HTTP {exc.code}", exc.code, payload) from exc
