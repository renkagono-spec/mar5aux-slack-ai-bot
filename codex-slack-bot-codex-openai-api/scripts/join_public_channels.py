from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.slack_client import SlackClient


def list_public_channels(slack_client: SlackClient, sleep_seconds: float) -> list[dict]:
    channels: list[dict] = []
    cursor = None

    while True:
        response = slack_client.conversation_list(
            types="public_channel",
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
    parser = argparse.ArgumentParser(description="Bulk-join public Slack channels with the bot token.")
    parser.add_argument("--all", action="store_true", help="Join every visible public channel")
    parser.add_argument("--prefix", action="append", default=[], help="Join public channels whose name starts with this prefix")
    parser.add_argument("--contains", action="append", default=[], help="Join public channels whose name contains this text")
    parser.add_argument("--channel-id", action="append", default=[], help="Join a specific public channel ID")
    parser.add_argument("--execute", action="store_true", help="Actually join channels. Without this, only prints a dry run")
    parser.add_argument("--sleep", type=float, default=1.2, help="Seconds to sleep between Slack API calls")
    args = parser.parse_args()

    if not (args.all or args.prefix or args.contains or args.channel_id):
        parser.error("Choose --all, --prefix, --contains, or --channel-id")

    settings = get_settings()
    slack_client = SlackClient(settings)
    channels = list_public_channels(slack_client, args.sleep)
    targets = [channel for channel in channels if channel_matches(channel, args)]

    if not targets:
        print("No matching public channels were found.")
        return

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: {len(targets)} public channels matched")

    joined = 0
    failed = 0
    for channel in targets:
        channel_id = channel.get("id")
        name = channel.get("name") or channel_id
        if not args.execute:
            print(f"would join #{name} ({channel_id})")
            continue

        try:
            slack_client.conversation_join(channel_id)
            joined += 1
            print(f"joined #{name} ({channel_id})")
        except Exception as exc:
            failed += 1
            print(f"failed #{name} ({channel_id}): {exc}")
        time.sleep(args.sleep)

    if args.execute:
        print(f"done: joined={joined} failed={failed}")


if __name__ == "__main__":
    main()
