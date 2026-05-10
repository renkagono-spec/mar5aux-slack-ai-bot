from __future__ import annotations

from functools import lru_cache
from typing import Any
from urllib.parse import urlencode

from .config import Settings
from .http_json import JsonApiError, get_json, post_json


class SlackClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._own_bot_id: str | None = settings.slack_own_bot_id
        self._own_user_id: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.slack_bot_token}"}

    def api_post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = post_json(
            f"https://slack.com/api/{method}",
            payload,
            headers=self.headers,
            timeout=45,
        )
        if not response.get("ok"):
            raise JsonApiError(f"Slack API {method} failed: {response.get('error')}", payload=response)
        return response

    def api_get(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        response = get_json(
            f"https://slack.com/api/{method}?{query}",
            headers=self.headers,
            timeout=45,
        )
        if not response.get("ok"):
            raise JsonApiError(f"Slack API {method} failed: {response.get('error')}", payload=response)
        return response

    def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        return self.api_post("chat.postMessage", payload)

    def add_reaction(self, channel: str, ts: str, name: str) -> None:
        try:
            self.api_post("reactions.add", {"channel": channel, "timestamp": ts, "name": name})
        except Exception:
            pass

    def get_permalink(self, channel: str, ts: str) -> str | None:
        try:
            response = self.api_get("chat.getPermalink", {"channel": channel, "message_ts": ts})
            return response.get("permalink")
        except Exception:
            return None

    @lru_cache(maxsize=2048)
    def channel_name(self, channel: str) -> str | None:
        try:
            response = self.api_get("conversations.info", {"channel": channel})
            info = response.get("channel") or {}
            return info.get("name")
        except Exception:
            return None

    @lru_cache(maxsize=4096)
    def user_name(self, user: str | None) -> str | None:
        if not user:
            return None
        try:
            response = self.api_get("users.info", {"user": user})
            info = response.get("user") or {}
            return info.get("profile", {}).get("display_name") or info.get("real_name") or info.get("name")
        except Exception:
            return None

    def own_bot_id(self) -> str | None:
        if self._own_bot_id:
            return self._own_bot_id
        try:
            response = self.auth_test()
            self._own_bot_id = response.get("bot_id")
            self._own_user_id = response.get("user_id")
            return self._own_bot_id
        except Exception:
            return None

    def own_user_id(self) -> str | None:
        if self._own_user_id:
            return self._own_user_id
        try:
            response = self.auth_test()
            self._own_bot_id = response.get("bot_id") or self._own_bot_id
            self._own_user_id = response.get("user_id")
            return self._own_user_id
        except Exception:
            return None

    def auth_test(self) -> dict[str, Any]:
        return self.api_post("auth.test", {})

    def conversation_history(
        self,
        channel: str,
        limit: int = 200,
        cursor: str | None = None,
        oldest: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "channel": channel,
            "limit": min(limit, 200),
        }
        if cursor:
            payload["cursor"] = cursor
        if oldest:
            payload["oldest"] = oldest
        return self.api_post("conversations.history", payload)

    def conversation_replies(
        self,
        channel: str,
        ts: str,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "channel": channel,
            "ts": ts,
            "limit": min(limit, 200),
        }
        if cursor:
            params["cursor"] = cursor
        return self.api_get("conversations.replies", params)
