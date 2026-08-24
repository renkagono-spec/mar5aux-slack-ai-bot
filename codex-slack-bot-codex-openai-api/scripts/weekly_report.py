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
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _report_common import (
    compact,
    message_post_date,
    most_populated_workspace_id,
    openai_complete,
    parse_json_list,
)
from slack_ai_bot.config import get_settings
from slack_ai_bot.search import JST, message_datetime_jst
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage, StoredMessage


# Single-day relative-time words and their day offset from the post date.
# Only unambiguous ones are listed: range words (今週/来週/月末) and 当日
# (refers to some event day, not the post day) are intentionally excluded so
# we never resolve them to a wrong date.
_RELATIVE_DAY_OFFSETS = {
    "本日": 0, "今日": 0, "今朝": 0, "今晩": 0, "今夜": 0,
    "明後日": 2, "明日": 1, "一昨日": -2, "昨日": -1,
}
# Longer words first so 明後日 / 一昨日 win over 明日 / 昨日. The lookahead skips
# words that already carry a (M/D) so re-runs stay idempotent.
_RELATIVE_DAY_RE = re.compile(
    "(" + "|".join(_RELATIVE_DAY_OFFSETS) + r")(?!\s*[（(])"
)


def annotate_relative_dates(text: str, post_date) -> str:
    """Append the absolute date to relative-day words, e.g. '本日' -> '本日(5/29)'.

    post_date is the message's own JST date, so each word resolves against the
    day it was written. This is done deterministically in code because the model
    cannot be trusted to do calendar math reliably.
    """
    if not text or post_date is None:
        return text

    def repl(match: "re.Match[str]") -> str:
        word = match.group(1)
        resolved = post_date + timedelta(days=_RELATIVE_DAY_OFFSETS[word])
        return f"{word}({resolved.month}/{resolved.day})"

    return _RELATIVE_DAY_RE.sub(repl, text)


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
    "SPECIFICITY (apply only WITHIN the factuality rules above; never override them):\n"
    "- Do NOT write vague summaries like '〜が共有された', '〜について話し合われた', or '〜が報告された' that omit the substance. "
    "State the actual substance instead.\n"
    "- For each bullet, prefer to include the concrete details the source literally provides: WHAT (品番・固有名詞・件名), "
    "WHICH (対象), HOW MANY / amount (数量・金額), WHEN (日付・期限), WHO/WHOM (相手・担当). "
    "Copy these values from the source as written.\n"
    "- If a given concrete value is NOT in the source, simply omit it. Never invent or estimate it to make a bullet look richer. "
    "Omitting an unknown detail is always correct; fabricating one is always wrong.\n"
    "- RELATIVE DATES are already resolved for you: the source text annotates words like 本日/今日/明日/明後日/昨日/一昨日 "
    "with their absolute date in parentheses, e.g. '本日(5/29)発送'. ALWAYS carry that '(M/D)' through verbatim into your "
    "bullet — never drop it and never alter the number. If a relative word has NO parenthesized date (e.g. 来週, 月末, 当日), "
    "it was deliberately left unresolved; keep it as written and do not guess a date for it.\n"
    "\n"
    "Merge duplicate facts across sources. Keep only what matters; drop trivia. Aim for at most about 6 bullets "
    "per section. If a section genuinely has nothing, write '• 特になし' under it. "
    "Do not add any other sections or preamble."
)

CITATION_RE = re.compile(r"\[(\d{1,3})\]")

# Mar5aux email-forwarding posts look like:
#   *From:* "name" <addr@example.com>
#   *件名:* Subject Line
# These regexes pull out those fields so the noise appendix is scannable.
MAIL_FROM_RE = re.compile(r"\*From:\*\s*(.+)")
MAIL_SUBJECT_RE = re.compile(r"\*件名:\*\s*(.+)")


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def extract_mail_summary(text: str) -> tuple[str, str]:
    """Pull a short (sender, subject) pair out of a forwarded-email post.

    Falls back to empty strings if the text is not an email-shaped post.
    """
    if not text:
        return "", ""
    sender = ""
    subject = ""
    m = MAIL_FROM_RE.search(text)
    if m:
        sender = m.group(1).strip()
    s = MAIL_SUBJECT_RE.search(text)
    if s:
        subject = s.group(1).strip()
    return sender, subject


def extract_sender_email(sender: str) -> str:
    """Pull the email address out of a 'From:' field for dedup purposes.

    Display names can be forged independently per message ('"c:/" <real@addr>',
    '"../" <real@addr>'), so dedup keyed on the display name lets the same
    attacker fill the noise list. The email address itself is more stable.
    """
    if not sender:
        return ""
    match = EMAIL_RE.search(sender)
    return match.group(0).lower() if match else sender.lower()


# Subject / body patterns that are very likely vulnerability scanning probes
# (path traversal, request for system files, raw URI references).  We surface
# these in their own "セキュリティ要注意" bucket so the user does not miss them
# inside the regular noise dump.
SUSPICIOUS_PATTERNS = (
    "/etc/passwd",
    "\\windows\\system",
    "/windows/system",
    "web-inf",
    "wp-config",
    "../../",
    "..\\..",
    ".env",
    "boot.ini",
    "shadow",
    "id_rsa",
)


def looks_suspicious(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in SUSPICIOUS_PATTERNS)


def _format_noise_line(message: StoredMessage) -> str:
    sender, subject = extract_mail_summary(message.text)
    compact_sender = compact(sender, 60) or "—"
    compact_subject = compact(subject, 80) or compact(message.text, 80)
    when = message_datetime_jst(message)[5:10]  # MM-DD
    channel = message.channel_name or message.channel_id
    link = f" <{message.permalink}|🔗開く>" if message.permalink else ""
    return f"• ({when} #{channel}) From: {compact_sender} / 件名: {compact_subject}{link}"


def format_noise_appendix(
    excluded: list[StoredMessage],
    suspicious_limit: int = 10,
) -> str:
    """Surface only suspicious / attack-shaped mail from the excluded set.

    The regular "营業・通知" noise bucket is intentionally not shown: the user
    asked to keep the report focused. We still surface forwarded mail that
    matches well-known attack reconnaissance patterns (path traversal, system
    file requests, common probe paths) because those are something the user
    almost certainly wants to know about.
    """
    if not excluded:
        return ""

    suspicious = [m for m in excluded if looks_suspicious(m.text)]
    if not suspicious:
        return ""

    suspicious = sorted(
        suspicious,
        key=lambda m: float(m.ts) if m.ts.replace(".", "").isdigit() else 0.0,
        reverse=True,
    )
    shown = suspicious[:suspicious_limit]

    lines: list[str] = [
        "",
        "*【⚠️ セキュリティ要注意（攻撃の可能性）】*",
        f"_お問い合わせフォーム等への自動脆弱性スキャン疑い（path traversal、システムファイル要求など）。"
        f"検知{len(suspicious)}件中{len(shown)}件を新しい順で表示。"
        f"内容を確認の上、必要なら送信元のブロックやフォームの入力サニタイズ確認を。_",
    ]
    for message in shown:
        lines.append(_format_noise_line(message))
    if len(suspicious) > suspicious_limit:
        lines.append(f"_他 {len(suspicious) - suspicious_limit} 件は省略_")

    return "\n".join(lines)

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
            parsed = parse_json_list(raw)
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


def build_report(settings, days: int, max_messages_per_channel: int,
                 own_user_id: str | None, max_total_sources: int = 260,
                 exclude_mail_noise: bool = True,
                 oldest_ts_override: str | None = None) -> tuple[str, dict]:
    storage = Storage(settings)
    storage.init_schema()

    workspace_id = most_populated_workspace_id(storage)
    if not workspace_id:
        return "対象ワークスペースのデータがありません。", {"channels": 0, "messages": 0}

    now = datetime.now(JST)
    if oldest_ts_override:
        # Continue from the previous report: cover exactly the span since it
        # was posted, so consecutive reports never overlap and never leave gaps.
        start = datetime.fromtimestamp(float(oldest_ts_override), JST)
        start_ts = oldest_ts_override
    else:
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
    # We keep the excluded items in a separate list so the report can show them
    # at the end as a "possibly-noise" appendix — the user wants to be able to
    # spot any real mail that the triage misclassified.
    excluded_noise = 0
    excluded_messages: list[StoredMessage] = []
    if exclude_mail_noise:
        bot_messages = [m for m in capped if m.source_type == "bot_message" and compact(m.text, 1)]
        if bot_messages:
            keep_keys = classify_keep_important(settings, bot_messages)
            filtered: list[StoredMessage] = []
            for message in capped:
                if message.source_type == "bot_message" and (message.channel_id, message.ts) not in keep_keys:
                    excluded_noise += 1
                    excluded_messages.append(message)
                    continue
                filtered.append(message)
            capped = filtered

    index_to_message: dict[int, StoredMessage] = {}
    source_lines: list[str] = []
    index = 0
    for message in capped:
        if index >= max_total_sources:
            break
        body_text = compact(message.text, 600)
        if not body_text:
            continue
        body_text = annotate_relative_dates(body_text, message_post_date(message))
        index += 1
        index_to_message[index] = message
        name = message.channel_name or message.channel_id
        who = message.user_name or ("メール" if message.source_type == "bot_message" else (message.user_id or "unknown"))
        when = message_datetime_jst(message)[:16]  # YYYY-MM-DD HH:MM
        source_lines.append(f"[{index}] (#{name}) {when} {who}: {body_text}")

    reduce_input = "\n".join(source_lines)[:48000]

    error: str | None = None
    if not reduce_input.strip():
        body = "今週は目立った決定事項・要フォロー・新規の動きはありませんでした。"
    else:
        try:
            raw = openai_complete(settings, REPORT_INSTRUCTIONS, reduce_input, timeout=150)
            body = linkify_citations(raw, index_to_message)
        except Exception as exc:  # noqa: BLE001
            body = f"レポート生成に失敗しました: {exc}"
            error = str(exc)

    noise_appendix = format_noise_appendix(excluded_messages)
    if noise_appendix:
        body = f"{body}\n\n{noise_appendix}"

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
        "error": error,
    }
    return f"{header}\n{body}", stats


# NOTE: Slack returns the posted 📊 emoji as ':bar_chart:' in history text, so
# the marker must not include the emoji itself.
REPORT_HEADER_MARK = "Slack週次レポート"


def find_last_report_ts(slack_client: SlackClient, channel: str) -> str | None:
    """Return the ts of the most recent report previously posted in `channel`.

    Reports are recognized by their fixed header mark, so Slack itself acts as
    the record of how far reporting has progressed — no local state file.
    """
    cursor = None
    for _ in range(5):  # up to ~1000 messages back
        response = slack_client.conversation_history(channel=channel, limit=200, cursor=cursor)
        for item in response.get("messages", []):  # newest first
            text = item.get("text") or ""
            # Only successful reports mark progress; skip failed-generation posts
            # so a broken run never becomes the anchor and swallows a week.
            if REPORT_HEADER_MARK in text[:80] and "レポート生成に失敗" not in text:
                return item.get("ts")
        cursor = (response.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a weekly Slack digest from the bot database.")
    parser.add_argument("--days", type=int, default=7, help="How many days back to include (default 7)")
    parser.add_argument("--max-messages-per-channel", type=int, default=80,
                        help="Cap messages per channel fed to the summarizer (most recent kept)")
    parser.add_argument("--keep-mail-noise", action="store_true",
                        help="Do NOT filter forwarded mail/automated posts (by default sales/notification/spam mail is dropped)")
    parser.add_argument("--post", action="store_true", help="Actually post to Slack (otherwise dry run / console only)")
    parser.add_argument("--channel", help="Target Slack channel ID for --post, e.g. C0AKHJTU2H2")
    parser.add_argument("--since-last", action="store_true",
                        help="Start from the previous report posted in --channel instead of --days back, "
                             "so consecutive reports never overlap. Falls back to --days if none is found.")
    args = parser.parse_args()

    settings = get_settings()
    slack_client = SlackClient(settings)

    own_user_id = None
    try:
        own_user_id = slack_client.own_user_id()
    except Exception:
        own_user_id = None

    oldest_ts_override = None
    if args.since_last:
        if not args.channel:
            print("--since-last には --channel が必要です(前回レポートを探すチャンネル)。")
            return
        oldest_ts_override = find_last_report_ts(slack_client, args.channel)
        if oldest_ts_override:
            since = datetime.fromtimestamp(float(oldest_ts_override), JST)
            print(f"[since-last] 前回レポート: {since.strftime('%Y-%m-%d %H:%M')} 以降を対象にします。")
        else:
            print(f"[since-last] 前回レポートが見つからないため --days {args.days} で実行します。")

    report, stats = build_report(
        settings,
        days=args.days,
        max_messages_per_channel=args.max_messages_per_channel,
        own_user_id=own_user_id,
        exclude_mail_noise=not args.keep_mail_noise,
        oldest_ts_override=oldest_ts_override,
    )

    print("=" * 60)
    print(report)
    print("=" * 60)
    print(f"[stats] {stats}")

    if args.post:
        if not args.channel:
            print("\n--post was given but --channel is missing. Nothing posted.")
            return
        if stats.get("error"):
            # Never post a broken report: leave Slack clean, exit non-zero so the
            # failure is visible and the next run resumes from the last good report.
            print(f"\n生成に失敗したため投稿をスキップしました: {stats['error']}")
            raise SystemExit(1)
        slack_client.post_message(channel=args.channel, text=report[:39000])
        print(f"\nPosted to channel {args.channel}.")
    else:
        print("\n(dry run) Re-run with --post --channel CXXXXXXX to post this to Slack.")


if __name__ == "__main__":
    main()
