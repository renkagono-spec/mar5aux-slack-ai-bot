from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.handlers import message_from_event
from slack_ai_bot.openai_client import OpenAIClient
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage


def backfill_channel(channel_id: str, max_messages: int, sleep_seconds: float) -> int:
    settings = get_settings()
    storage = Storage(settings)
    slack_client = SlackClient(settings)
    openai_client = OpenAIClient(settings)

    storage.init_schema()

    cursor = None
    stored = 0
    payload_base = {"team_id": slack_client.auth_test().get("team_id") or "unknown-workspace"}

    while stored < max_messages:
        response = slack_client.conversation_history(channel=channel_id, limit=min(200, max_messages - stored), cursor=cursor)
        messages = response.get("messages") or []
        for event in messages:
            event.setdefault("channel", channel_id)
            message = message_from_event(payload_base, event, slack_client, openai_client, settings)
            if message:
                storage.upsert_message(message, raw=event)
                stored += 1

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor or not messages:
            break
        time.sleep(sleep_seconds)

    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Slack channel history into the bot database.")
    parser.add_argument("channels", nargs="+", help="Slack channel IDs, for example C12345678")
    parser.add_argument("--max-messages", type=int, default=500, help="Maximum messages per channel")
    parser.add_argument("--sleep", type=float, default=1.2, help="Seconds to sleep between Slack API pages")
    args = parser.parse_args()

    total = 0
    for channel in args.channels:
        count = backfill_channel(channel, args.max_messages, args.sleep)
        print(f"{channel}: stored {count} messages")
        total += count
    print(f"done: stored {total} messages")


if __name__ == "__main__":
    main()
