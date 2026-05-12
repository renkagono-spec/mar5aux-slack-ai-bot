from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage


try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9), name="JST")


def ts_to_jst(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(value), JST).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "-"


def list_visible_joined_public_channels(slack_client: SlackClient) -> dict[str, str]:
    channels: dict[str, str] = {}
    cursor = None
    while True:
        response = slack_client.conversation_list(
            types="public_channel",
            limit=1000,
            cursor=cursor,
            exclude_archived=True,
        )
        for channel in response.get("channels") or []:
            if channel.get("is_member"):
                channels[channel["id"]] = channel.get("name") or channel["id"]
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Show how much Slack data is indexed in the bot database.")
    parser.add_argument("--workspace-id", help="Only report one workspace ID")
    parser.add_argument("--limit", type=int, default=50, help="Maximum DB channels to print")
    parser.add_argument("--check-slack", action="store_true", help="Also compare against visible joined public Slack channels")
    args = parser.parse_args()

    settings = get_settings()
    storage = Storage(settings)
    stats = storage.channel_message_stats(workspace_id=args.workspace_id)

    print(f"indexed_channels={len(stats)}")
    print("channel_name\tchannel_id\tmessages\treplies\toldest_jst\tlatest_jst\tmissing_embeddings")
    for row in stats[: args.limit]:
        print(
            "\t".join(
                [
                    str(row.get("channel_name") or ""),
                    str(row.get("channel_id") or ""),
                    str(row.get("message_count") or 0),
                    str(row.get("reply_count") or 0),
                    ts_to_jst(row.get("oldest_ts")),
                    ts_to_jst(row.get("latest_ts")),
                    str(row.get("missing_embedding_count") or 0),
                ]
            )
        )

    if not args.check_slack:
        return

    slack_client = SlackClient(settings)
    joined = list_visible_joined_public_channels(slack_client)
    indexed_ids = {row["channel_id"] for row in stats}
    missing = [(channel_id, name) for channel_id, name in joined.items() if channel_id not in indexed_ids]
    print("")
    print(f"joined_public_channels_visible={len(joined)}")
    print(f"joined_public_channels_missing_from_db={len(missing)}")
    for channel_id, name in sorted(missing, key=lambda item: item[1]):
        print(f"missing\t#{name}\t{channel_id}")


if __name__ == "__main__":
    main()
