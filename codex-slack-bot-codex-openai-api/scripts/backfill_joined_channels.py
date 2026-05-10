from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backfill import backfill_channel
from slack_ai_bot.config import get_settings
from slack_ai_bot.slack_client import SlackClient


def list_channels(slack_client: SlackClient, channel_types: str, sleep_seconds: float) -> list[dict]:
    channels: list[dict] = []
    cursor = None

    while True:
        response = slack_client.conversation_list(
            types=channel_types,
            limit=1000,
            cursor=cursor,
            exclude_archived=True,
        )
        channels.extend(response.get("channels") or [])
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_seconds)

    return channels


def channel_matches(channel: dict, args: argparse.Namespace) -> bool:
    channel_id = channel.get("id") or ""
    name = channel.get("name") or ""

    if args.channel_id and channel_id in set(args.channel_id):
        return True
    if args.all:
        return True
    if args.prefix and any(name.startswith(prefix) for prefix in args.prefix):
        return True
    if args.contains and any(fragment in name for fragment in args.contains):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill every matching channel the bot is already a member of.")
    parser.add_argument("--all", action="store_true", help="Backfill every joined visible channel")
    parser.add_argument("--prefix", action="append", default=[], help="Backfill joined channels whose name starts with this prefix")
    parser.add_argument("--contains", action="append", default=[], help="Backfill joined channels whose name contains this text")
    parser.add_argument("--channel-id", action="append", default=[], help="Backfill a specific channel ID")
    parser.add_argument(
        "--types",
        default="public_channel",
        help="Slack conversation types to scan. Use private_channel only when the app has private channel scopes.",
    )
    parser.add_argument("--max-messages", type=int, default=1000, help="Maximum parent messages per channel")
    parser.add_argument("--sleep", type=float, default=1.2, help="Seconds to sleep between Slack API pages")
    parser.add_argument("--thread-sleep", type=float, default=1.2, help="Seconds to sleep between thread API calls")
    parser.add_argument("--thread-page-limit", type=int, default=200, help="Maximum replies to fetch per thread API page")
    parser.add_argument("--no-threads", action="store_true", help="Only ingest channel history, without thread replies")
    parser.add_argument("--execute", action="store_true", help="Actually backfill. Without this, only prints a dry run")
    args = parser.parse_args()

    if not (args.all or args.prefix or args.contains or args.channel_id):
        parser.error("Choose --all, --prefix, --contains, or --channel-id")

    settings = get_settings()
    slack_client = SlackClient(settings)
    channels = list_channels(slack_client, args.types, args.sleep)
    targets = [
        channel
        for channel in channels
        if channel.get("is_member") and channel_matches(channel, args)
    ]

    if not targets:
        print("No matching joined channels were found.")
        return

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: {len(targets)} joined channels matched")

    total = 0
    for channel in targets:
        channel_id = channel["id"]
        name = channel.get("name") or channel_id
        if not args.execute:
            print(f"would backfill #{name} ({channel_id})")
            continue

        try:
            count = backfill_channel(
                channel_id=channel_id,
                max_messages=args.max_messages,
                sleep_seconds=args.sleep,
                include_threads=not args.no_threads,
                thread_sleep_seconds=args.thread_sleep,
                thread_page_limit=args.thread_page_limit,
            )
            total += count
            print(f"backfilled #{name} ({channel_id}): stored {count} messages")
        except Exception as exc:
            print(f"failed #{name} ({channel_id}): {exc}")
        time.sleep(args.sleep)

    if args.execute:
        print(f"done: stored {total} messages")


if __name__ == "__main__":
    main()
