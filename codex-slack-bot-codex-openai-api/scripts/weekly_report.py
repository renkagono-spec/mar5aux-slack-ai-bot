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
import json
from pathlib import Path
import re
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
    "You write a concise internal weekly Slack report in Japanese from numbered source messages. "
    "Each source line is formatted as '[n] (#channel) date who: text'. "
    "Output EXACTLY these three Slack headings in this order: "
    "*【主な決定事項】*, *【未対応・要フォロー】*, *【新規の動き】*. "
    "Under each, write short bullet points starting with '• '. "
    "\n\n"
    "STRICT FACTUALITY RULES (these override every other instruction):\n"
    "- Use ONLY facts explicitly written in the sources. Never infer, guess, extrapolate, or invent.\n"
    "- NEVER fabricate or estimate dates, times, schedules, deadlines, amounts, next actions, or status "
    "changes. If a source does not literally state it, do not write it. (For example: do not write "
    "'5/31に現地確認予定' unless a source literally says so.)\n"
    "- Do NOT claim something is 完了 / 入金済み / 予定 / 決定 unless a source explicitly says so. If a status "
    "is unclear or only partly stated, describe only what is literally written and stop there.\n"
    "- Every bullet MUST end with the source number(s) it is based on, like [12][34], using ONLY numbers that "
    "exist in the input. If you cannot cite a real source number for a statement, DELETE that statement.\n"
    "- Each bullet must be supported by its cited source(s) ALONE. Do not combine separate sources into a "
    "claim that none of them makes on its own.\n"
    "- When in doubt, write less. A short fully-grounded report is correct; a richer but inferred one is wrong.\n"
    "\n"
    "Merge duplicate facts across sources. Keep only what matters; drop trivia. Aim for at most about 6 bullets "
    "per section. If a section genuinely has nothing, write '• 特になし' under it. "
    "Do not add any other sections or preamble."
)

CITATION_RE = re.compile(r"\[(\d{1,3})\]")

TRIAGE_INSTRUCTIONS = (
    "You triage forwarded emails and automated Slack messages for an apparel company. "
    "Each numbered item is one message. Return JSON only: a list of objects {\"n\": int, \"keep\": bool}. "
    "keep=true ONLY when the message is a real, person-to-person business matter that a human must act on: "
    "orders, quotes, invoices/payments, production or delivery scheduling, customer or partner replies, "
    "meeting scheduling, or anything affecting a live deal. "
    "keep=false for newsletters, ad/marketing mail, seminar or event invitations, cold sales pitches, "
    "system notifications, receipts, automatic delivery/shipping notices, and spam. "
    "When genuinely unsure, keep=true (do not drop a possibly-important business mail). Return only the JSON list."
)


def parse_json_object_list(raw: str) -> list:
    """Parse a JSON list of objects, tolerating models that omit the [] wrapper.

    The model sometimes returns ``{"n":1},{"n":2}`` (comma-separated objects with
    no array brackets). Wrap those before parsing so triage decisions are not lost.
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


def classify_keep_important(settings, messages: list[StoredMessage], batch_size: int = 40) -> set[tuple[str, str]]:
    """Return the set of (channel_id, ts) for bot/email messages worth keeping.

    On any failure we fail open (treat the batch as keep) so the report never
    silently loses real business mail because of a transient API issue.
    """
    keep: set[tuple[str, str]] = set()
    for start in range(0, len(messages), batch_size):
        batch = messages[start:start + batch_size]
        lines = [f"[{i + 1}] {compact(m.text, 400)}" for i, m in enumerate(batch)]
        try:
            raw = openai_complete(settings, TRIAGE_INSTRUCTIONS, "\n\n".join(lines), timeout=120)
            parsed = parse_json_object_list(raw)
            if not parsed:
                raise ValueError("empty triage result")
            decided: dict[int, bool] = {}
            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                n = obj.get("n")
                if isinstance(n, int) and 1 <= n <= len(batch):
                    decided[n] = bool(obj.get("keep"))
            for i, message in enumerate(batch, start=1):
                if decided.get(i, True):  # default keep when the model omitted an item
                    keep.add((message.channel_id, message.ts))
        except Exception:  # noqa: BLE001 - fail open
            for message in batch:
                keep.add((message.channel_id, message.ts))
    return keep


def linkify_citations(report_text: str, index_to_message: dict[int, StoredMessage], max_links: int = 2) -> str:
    """Replace trailing [n] markers on each line with compact Slack permalinks.

    Each citation becomes a short numbered link like ``[1]`` that points to the
    source message, with the channel name on hover, so the bullet text stays
    readable instead of being buried under full URLs.
    """
    out_lines: list[str] = []
    for line in report_text.split("\n"):
        numbers = [int(value) for value in CITATION_RE.findall(line)]
        if not numbers:
            out_lines.append(line)
            continue

        cleaned = CITATION_RE.sub("", line).rstrip()
        links: list[str] = []
        seen: set[str] = set()
        for position, number in enumerate(numbers, start=1):
            message = index_to_message.get(number)
            if not message or not message.permalink or message.permalink in seen:
                continue
            seen.add(message.permalink)
            channel = f"#{message.channel_name}" if message.channel_name else message.channel_id
            when = message_datetime_jst(message)[5:10]  # MM-DD
            # tooltip text shows channel + date; visible label stays tiny.
            links.append(f"<{message.permalink}|🔗{channel} {when}>")
            if len(links) >= max_links:
                break

        if links:
            out_lines.append(f"{cleaned}  " + " ".join(links))
        else:
            out_lines.append(cleaned)
    return "\n".join(out_lines)


def build_report(settings, days: int, min_channel_messages: int, max_messages_per_channel: int,
                 own_user_id: str | None, max_total_sources: int = 260,
                 exclude_mail_noise: bool = True) -> tuple[str, dict]:
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

    # Cap each channel to its most recent N messages, then merge everything into
    # one workspace-wide chronological stream. Capping per channel first stops a
    # single high-volume mail channel from crowding out the human-conversation
    # channels; the chronological merge keeps related context near each other and
    # spreads source numbers across all channels so citations are not lopsided.
    capped: list[StoredMessage] = []
    for channel_messages in by_channel.values():
        channel_messages.sort(key=lambda m: float(m.ts) if m.ts.replace(".", "").isdigit() else 0.0)
        capped.extend(channel_messages[-max_messages_per_channel:])
    capped.sort(key=lambda m: float(m.ts) if m.ts.replace(".", "").isdigit() else 0.0)

    # Drop noise from forwarded mail / automated posts: keep human (slack_message)
    # messages always, but for bot_message items keep only the ones AI triage
    # marks as real business matters (orders, invoices, scheduling, replies).
    excluded_noise = 0
    if exclude_mail_noise:
        bot_messages = [m for m in capped if m.source_type == "bot_message" and compact(m.text, 1)]
        if bot_messages:
            keep_keys = classify_keep_important(settings, bot_messages)
            filtered: list[StoredMessage] = []
            for message in capped:
                if message.source_type == "bot_message" and (message.channel_id, message.ts) not in keep_keys:
                    excluded_noise += 1
                    continue
                filtered.append(message)
            capped = filtered

    index_to_message: dict[int, StoredMessage] = {}
    source_lines: list[str] = []
    index = 0
    for message in capped:
        if index >= max_total_sources:
            break
        body_text = compact(message.text, 300)
        if not body_text:
            continue
        index += 1
        index_to_message[index] = message
        name = message.channel_name or message.channel_id
        who = message.user_name or ("メール" if message.source_type == "bot_message" else (message.user_id or "unknown"))
        when = message_datetime_jst(message)[:16]  # YYYY-MM-DD HH:MM
        source_lines.append(f"[{index}] (#{name}) {when} {who}: {body_text}")

    reduce_input = "\n".join(source_lines)[:48000]

    if not reduce_input.strip():
        body = "今週は目立った決定事項・要フォロー・新規の動きはありませんでした。"
    else:
        try:
            raw = openai_complete(settings, REPORT_INSTRUCTIONS, reduce_input, timeout=150)
            body = linkify_citations(raw, index_to_message)
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
        "sources_used": index,
        "excluded_noise": excluded_noise,
    }
    return f"{header}\n{body}", stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a weekly Slack digest from the bot database.")
    parser.add_argument("--days", type=int, default=7, help="How many days back to include (default 7)")
    parser.add_argument("--min-channel-messages", type=int, default=4,
                        help="Channels with at least this many messages get an AI summary; smaller ones are listed raw")
    parser.add_argument("--max-messages-per-channel", type=int, default=80,
                        help="Cap messages per channel fed to the summarizer (most recent kept)")
    parser.add_argument("--keep-mail-noise", action="store_true",
                        help="Do NOT filter forwarded mail/automated posts (by default sales/notification/spam mail is dropped)")
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
        exclude_mail_noise=not args.keep_mail_noise,
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
