"""Shared helpers for the two surviving automation scripts:
weekly_report.py and invoice_reminder.py.

Both read the bot's database (populated by the running Slack bot's message
ingestion) and call the OpenAI Responses API, so the small bits of glue they
both need live here instead of being copy-pasted.
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from slack_ai_bot.http_json import JsonApiError, post_json
from slack_ai_bot.search import message_datetime_jst
from slack_ai_bot.storage import Storage, StoredMessage

__all__ = [
    "JsonApiError",
    "compact",
    "openai_complete",
    "parse_json_list",
    "most_populated_workspace_id",
    "message_post_date",
]


def compact(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to `limit` chars with an ellipsis."""
    collapsed = " ".join((text or "").split())
    return collapsed[:limit] + ("…" if len(collapsed) > limit else "")


def openai_complete(settings, instructions: str, content: str, timeout: int = 120) -> str:
    """Call the OpenAI Responses API and return the text output (temperature 0)."""
    response = post_json(
        "https://api.openai.com/v1/responses",
        {
            "model": settings.openai_model,
            "instructions": instructions,
            "input": content,
            "temperature": 0,
        },
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        timeout=timeout,
    )
    text = response.get("output_text") or ""
    if not text:
        chunks: list[str] = []
        for item in response.get("output", []):
            for piece in item.get("content", []):
                if piece.get("type") == "output_text" and piece.get("text"):
                    chunks.append(piece["text"])
        text = "\n".join(chunks)
    return text.strip()


def parse_json_list(raw: str) -> list:
    """Parse a JSON list, tolerating models that omit the surrounding [].

    Some models return ``{"n":1},{"n":2}`` (comma-separated objects with no
    array brackets); wrap those before parsing so nothing is silently lost.
    """
    text = (raw or "").strip()
    if "[" in text and "]" in text and text.find("[") < text.rfind("]"):
        snippet = text[text.find("["): text.rfind("]") + 1]
    elif "{" in text and "}" in text:
        snippet = "[" + text[text.find("{"): text.rfind("}") + 1] + "]"
    else:
        return []
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def most_populated_workspace_id(storage: Storage) -> str | None:
    """Return the workspace id with the most stored messages (the live one)."""
    deleted = "FALSE" if storage.backend == "postgres" else "0"
    with storage.connect() as conn:
        row = conn.execute(
            f"SELECT workspace_id FROM messages WHERE is_deleted = {deleted} "
            "GROUP BY workspace_id ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
    return dict(row)["workspace_id"] if row else None


def message_post_date(message: StoredMessage) -> date | None:
    """The message's JST calendar date (a date object), or None if unparseable."""
    stamp = message_datetime_jst(message)[:10]  # YYYY-MM-DD
    try:
        return datetime.strptime(stamp, "%Y-%m-%d").date()
    except ValueError:
        return None
