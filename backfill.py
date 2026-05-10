from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.handlers import message_from_event
from slack_ai_bot.openai_client import OpenAIClient
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage


def store_message(
    payload_base: dict[str, str],
    event: dict[str, Any],
    channel_id: str,
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Any,
) -> bool:
    event.setdefault("channel", channel_id)
    message = message_from_event(payload_base, event, slack_client, openai_client, settings)
    if not message:
        return False
    storage.upsert_message(message, raw=event)
    return True


def has_thread_replies(event: dict[str, Any]) -> bool:
    reply_count = event.get("reply_count") or 0
    return bool(event.get("ts") and int(reply_count) > 0)


def backfill_thread(
    channel_id: str,
    parent_ts: str,
    payload_base: dict[str, str],
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Any,
    sleep_seconds: float,
    thread_page_limit: int,
) -> int:
    cursor = None
    stored = 0

    while True:
        response = slack_client.conversation_replies(
            channel=channel_id,
            ts=parent_ts,
            limit=thread_page_limit,
            cursor=cursor,
        )
        messages = response.get("messages") or []
        for event in messages:
            if store_message(payload_base, event, channel_id, storage, slack_client, openai_client, settings):
                stored += 1

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor or not messages:
            break
        time.sleep(sleep_seconds)

    return stored


def backfill_channel(
    channel_id: str,
    max_messages: int,
    sleep_seconds: float,
    include_threads: bool,
    thread_sleep_seconds: float,
    thread_page_limit: int,
) -> int:
    settings = get_settings()
    storage = Storage(settings)
    slack_client = SlackClient(settings)
    openai_client = OpenAIClient(settings)

    storage.init_schema()

    cursor = None
    stored = 0
    parent_messages_seen = 0
    payload_base = {"team_id": slack_client.auth_test().get("team_id") or "unknown-workspace"}

    while parent_messages_seen < max_messages:
        page_limit = min(200, max_messages - parent_messages_seen)
        response = slack_client.conversation_history(channel=channel_id, limit=page_limit, cursor=cursor)
        messages = response.get("messages") or []
        for event in messages:
            parent_messages_seen += 1
            if include_threads and has_thread_replies(event):
                stored += backfill_thread(
                    channel_id=channel_id,
                    parent_ts=event["ts"],
                    payload_base=payload_base,
                    storage=storage,
                    slack_client=slack_client,
                    openai_client=openai_client,
                    settings=settings,
                    sleep_seconds=thread_sleep_seconds,
                    thread_page_limit=thread_page_limit,
                )
                time.sleep(thread_sleep_seconds)
            elif store_message(payload_base, event, channel_id, storage, slack_client, openai_client, settings):
                stored += 1

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor or not messages or parent_messages_seen >= max_messages:
            break
        time.sleep(sleep_seconds)

    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Slack channel history into the bot database.")
    parser.add_argument("channels", nargs="+", help="Slack channel IDs, for example C12345678")
    parser.add_argument("--max-messages", type=int, default=500, help="Maximum parent messages per channel")
    parser.add_argument("--sleep", type=float, default=1.2, help="Seconds to sleep between Slack API pages")
    parser.add_argument("--no-threads", action="store_true", help="Only ingest channel history, without thread replies")
    parser.add_argument("--thread-sleep", type=float, default=1.2, help="Seconds to sleep between thread API calls")
    parser.add_argument("--thread-page-limit", type=int, default=200, help="Maximum replies to fetch per thread API page")
    args = parser.parse_args()

    total = 0
    for channel in args.channels:
        count = backfill_channel(
            channel,
            args.max_messages,
            args.sleep,
            include_threads=not args.no_threads,
            thread_sleep_seconds=args.thread_sleep,
            thread_page_limit=args.thread_page_limit,
        )
        print(f"{channel}: stored {count} messages")
        total += count
    print(f"done: stored {total} messages")


if __name__ == "__main__":
    main()
