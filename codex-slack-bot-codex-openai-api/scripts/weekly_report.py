"""Weekly Slack digest for the mar5aux Slack AI bot.

Pulls the last N days of messages from the bot's database, summarizes each
active channel with the OpenAI model (map), then combines everything into a
single weekly report with three sections (reduce):

    [主な決定事項]   key decisions made this week
    [未対応・要フォロー] open / unhandled items needing follow-up
    [新規の動き]      new topics, deals, or relationships that emerged

By default this is a DRY RUN: it only prints to the console. To actually post
to Slack you must pass both --post and --channel CXXXXXXX.

Examples:
    python scripts/weekly_report.py
    python scripts/weekly_report.py --days 7
    python scripts/weekly_report.py --post --channel C0AKHJTU2H2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.http_json import post_json
from slack_ai_bot.search import JST, message_datetime_jst
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage, StoredMessage


def most_populated_workspace_id(storage: Storage) -> str | None:
    with storage.connect() as conn:
        if storage.backend == "postgres":
            row = conn.execute(
                "SELECT workspace_id FROM messages WHERE is_deleted = FALSE "
                "GROUP BY workspace_id ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT workspace_id FROM messages WHERE is_deleted = 0 "
                "GROUP BY workspace_id ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
    return dict(row)["workspace_id"] if row else None


def openai_complete(settings, instructions: str, content: str, timeout: int = 120) -> str:
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


def compact(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit] + ("…" if len(collapsed) > limit else "")


def transcript_lines(messages: list[StoredMessage], per_message_chars: int) -> list[str]:
    lines: list[str] = []
    for message in messages:
        who = message.user_name or message.user_id or "unknown"
        body = compact(message.text, per_message_chars)
        if not body:
            continue
        lines.append(f"{message_datetime_jst(message)} {who}: {body}")
    return lines


CHANNEL_INSTRUCTIONS = (
    "You summarize one week of a single Slack channel for an internal weekly report. "
    "Answer in Japanese, factually, with no invented details. "
    "Extract only what is actually present, in short bullet points under three labels: "
    "決定 (decisions actually made), 未対応 (open questions, awaited replies, undecided or needs-confirmation items), "
    "新規 (new topics, deals, customers, or relationships that first appeared). "
    "If a label has nothing, omit it. If the whole channel has nothing notable, reply exactly with 特になし. "
    "Keep it concise: at most 5 bullets total."
)

REPORT_INSTRUCTIONS = (
    "You write a concise internal weekly Slack report in Japanese from per-channel summaries. "
    "Output EXACTLY these three sections in this order, each as a Slack-formatted heading: "
    "*【主な決定事項】*, *【未対応・要フォロー】*, *【新規の動き】*. "
    "Under each, write short bullet points starting with '• '. "
    "Put the source channel in parentheses at the end of each bullet, e.g. (#channel-name). "
    "Merge duplicates across channels and keep only what matters. Be factual, do not invent. "
    "If a section genuinely has nothing, write '• 特になし' under it. "
    "Do not add any other sections or preamble."
)


def build_report(settings, days: int, min_channel_messages: int, max_messages_per_channel: int,
                 own_user_id: str | None) -> tuple[str, dict]:
    storage = Storage(settings)
    storage.init_schema()

    workspace_id = most_populated_workspace_id(storage)
    if not workspace_id:
        return "対象ワークスペースのデータがありません。", {"channels": 0, "messages": 0}

    now = datetime.now(JST)
    start = now - timedelta(days=days)
    start_ts = str(start.timestamp())
    end_ts = str(now.timestamp())

    messages = storage.list_messages(
        workspace_id=workspace_id,
        channel_id=None,
        search_scope="workspace",
        limit=5000,
        oldest_ts=start_ts,
        latest_ts=end_ts,
    )

    # Drop questions addressed to our own bot (noise, not real content).
    if own_user_id:
        mention = f"<@{own_user_id}>"
        messages = [m for m in messages if mention not in (m.text or "")]

    if not messages:
        period = f"{start.strftime('%-m/%-d') if hasattr(start, 'strftime') else start}"
        return "今週(対象期間)に取り込まれたメッセージはありませんでした。", {"channels": 0, "messages": 0}

    by_channel: dict[str, list[StoredMessage]] = defaultdict(list)
    for message in messages:
        by_channel[message.channel_id].append(message)

    channel_summaries: list[str] = []
    minor_lines: list[str] = []
    summarized = 0

    # Larger channels first so the most active context leads the reduce input.
    for channel_id, channel_messages in sorted(by_channel.items(), key=lambda kv: len(kv[1]), reverse=True):
        channel_messages.sort(key=lambda m: float(m.ts) if m.ts.replace(".", "").isdigit() else 0.0)
        name = channel_messages[0].channel_name or channel_id

        if len(channel_messages) >= min_channel_messages:
            capped = channel_messages[-max_messages_per_channel:]
            transcript = "\n".join(transcript_lines(capped, per_message_chars=400))[:12000]
            try:
                digest = openai_complete(
                    settings,
                    CHANNEL_INSTRUCTIONS,
                    f"channel: #{name}\nmessages this week: {len(channel_messages)}\n\n{transcript}",
                )
            except Exception as exc:  # noqa: BLE001
                digest = f"(要約失敗: {exc})"
            if digest and digest.strip() != "特になし":
                channel_summaries.append(f"## #{name} ({len(channel_messages)}件)\n{digest}")
                summarized += 1
        else:
            for line in transcript_lines(channel_messages, per_message_chars=200):
                minor_lines.append(f"(#{name}) {line}")

    reduce_parts: list[str] = []
    if channel_summaries:
        reduce_parts.append("\n\n".join(channel_summaries))
    if minor_lines:
        reduce_parts.append("## その他の小さな動き\n" + "\n".join(minor_lines[:120]))
    reduce_input = "\n\n".join(reduce_parts)[:45000]

    if not reduce_input.strip():
        body = "今週は目立った決定事項・要フォロー・新規の動きはありませんでした。"
    else:
        try:
            body = openai_complete(settings, REPORT_INSTRUCTIONS, reduce_input, timeout=150)
        except Exception as exc:  # noqa: BLE001
            body = f"レポート生成に失敗しました: {exc}"

    header = (
        f"*📊 Slack週次レポート* "
        f"({start.strftime('%m/%d')}〜{now.strftime('%m/%d')})\n"
        f"対象: {len(by_channel)}チャンネル / {len(messages)}メッセージ\n"
    )
    stats = {
        "channels": len(by_channel),
        "messages": len(messages),
        "summarized_channels": summarized,
    }
    return f"{header}\n{body}", stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a weekly Slack digest from the bot database.")
    parser.add_argument("--days", type=int, default=7, help="How many days back to include (default 7)")
    parser.add_argument("--min-channel-messages", type=int, default=4,
                        help="Channels with at least this many messages get an AI summary; smaller ones are listed raw")
    parser.add_argument("--max-messages-per-channel", type=int, default=80,
                        help="Cap messages per channel fed to the summarizer (most recent kept)")
    parser.add_argument("--post", action="store_true", help="Actually post to Slack (otherwise dry run / console only)")
    parser.add_argument("--channel", help="Target Slack channel ID for --post, e.g. C0AKHJTU2H2")
    args = parser.parse_args()

    settings = get_settings()

    own_user_id = None
    try:
        own_user_id = SlackClient(settings).own_user_id()
    except Exception:
        own_user_id = None

    report, stats = build_report(
        settings,
        days=args.days,
        min_channel_messages=args.min_channel_messages,
        max_messages_per_channel=args.max_messages_per_channel,
        own_user_id=own_user_id,
    )

    print("=" * 60)
    print(report)
    print("=" * 60)
    print(f"[stats] {stats}")

    if args.post:
        if not args.channel:
            print("\n--post was given but --channel is missing. Nothing posted.")
            return
        slack_client = SlackClient(settings)
        slack_client.post_message(channel=args.channel, text=report[:39000])
        print(f"\nPosted to channel {args.channel}.")
    else:
        print("\n(dry run) Re-run with --post --channel CXXXXXXX to post this to Slack.")


if __name__ == "__main__":
    main()
