from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request


class JsonApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def read_error_payload(exc: error.HTTPError) -> Any:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def retry_delay(exc: error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 1.0), 60.0)
        except ValueError:
            pass
    return min(2.0**attempt, 30.0)


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    max_retries: int = 3,
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

    for attempt in range(max_retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            payload = read_error_payload(exc)
            if exc.code == 429 and attempt < max_retries:
                time.sleep(retry_delay(exc, attempt))
                continue
            raise JsonApiError(f"POST {url} failed with HTTP {exc.code}", exc.code, payload) from exc
        except error.URLError as exc:
            if attempt < max_retries:
                time.sleep(min(2.0**attempt, 30.0))
                continue
            raise JsonApiError(f"POST {url} failed: {exc.reason}") from exc

    raise JsonApiError(f"POST {url} failed")


def get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict[str, Any]:
    req = request.Request(url, method="GET", headers=headers or {})
    for attempt in range(max_retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            payload = read_error_payload(exc)
            if exc.code == 429 and attempt < max_retries:
                time.sleep(retry_delay(exc, attempt))
                continue
            raise JsonApiError(f"GET {url} failed with HTTP {exc.code}", exc.code, payload) from exc
        except error.URLError as exc:
            if attempt < max_retries:
                time.sleep(min(2.0**attempt, 30.0))
                continue
            raise JsonApiError(f"GET {url} failed: {exc.reason}") from exc

    raise JsonApiError(f"GET {url} failed")
