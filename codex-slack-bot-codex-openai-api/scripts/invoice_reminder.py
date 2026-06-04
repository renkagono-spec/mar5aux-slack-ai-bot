"""On-demand invoice payment-due reminder for the mar5aux Slack AI bot.

Scans the invoice channel (#all-1_請求書) in the bot's database, asks the model
to extract structured invoice records (vendor / amount / payment-due date /
paid-status), and for every invoice whose deadline is the target day (by default
*tomorrow* — i.e. a day-before reminder) posts a reminder REPLY in that
invoice's own Slack thread.

This is run MANUALLY, not on a schedule: you trigger it, it checks, and only if
there is a matching invoice does it remind.

By default this is a DRY RUN: it only prints to the console. To actually post to
Slack you must pass --post.

Anti-fabrication: a reminder is only ever produced from a payment deadline that
is literally written in a source message. If no deadline is stated, that invoice
is skipped — the model is told never to invent or guess a date.

Examples:
    python scripts/invoice_reminder.py                 # dry run, due tomorrow
    python scripts/invoice_reminder.py --due-in-days 0 # due today
    python scripts/invoice_reminder.py --post          # really post reminders
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
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

INVOICE_CHANNEL_ID = "C0AQA1P45LG"  # #all-1_請求書

_JP_WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


def compact(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit] + ("…" if len(collapsed) > limit else "")


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


def parse_json_list(raw: str) -> list:
    """Parse a JSON list, tolerating models that omit the [] wrapper."""
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


def message_post_date(message: StoredMessage) -> date | None:
    stamp = message_datetime_jst(message)[:10]  # YYYY-MM-DD
    try:
        return datetime.strptime(stamp, "%Y-%m-%d").date()
    except ValueError:
        return None


INVOICE_INSTRUCTIONS = (
    "You extract invoice / payment-due records from Slack messages (forwarded emails and human notes) "
    "in an apparel company's billing channel. Each numbered source line is formatted as "
    "'[n] (posted YYYY-MM-DD) text'. Return JSON ONLY: a list of objects, each "
    "{\"n\": int, \"vendor\": string, \"amount\": string, \"due_date\": string-or-null, \"status\": string}.\n"
    "RULES (strict, never invent anything):\n"
    "- One source line may list SEVERAL invoices (e.g. a list of unpaid companies). Output one object per "
    "(vendor, deadline) pair, all sharing that same n.\n"
    "- vendor: the company/person to be paid (or who issued the invoice), copied as written.\n"
    "- amount: the payable amount exactly as written, keeping the 円/¥/comma formatting. Use \"\" if none is stated.\n"
    "- due_date: the payment deadline as an ABSOLUTE date 'YYYY-MM-DD'. If the text gives only 'M/D' or '〇月末' "
    "etc., resolve it using THAT line's post date and the post date's year. If NO payment deadline is literally "
    "stated for that invoice, set due_date to null. NEVER guess, estimate, or invent a deadline.\n"
    "- status: '入金済み' if the message clearly states it was paid/received/消込; '未入金' if it clearly states "
    "unpaid / 未入金 / お振込みのお願い / 未着金; otherwise '不明'.\n"
    "- Include ONLY real invoices or payment requests with a payable amount or a clear deadline. Skip everything "
    "else (chit-chat, shipping notices, ads, system mail).\n"
    "Return only the JSON list, no prose."
)


def extract_invoices(settings, messages: list[StoredMessage], max_sources: int = 200):
    """Return (records, index_to_message). Each record: dict with n/vendor/amount/due_date/status."""
    index_to_message: dict[int, StoredMessage] = {}
    source_lines: list[str] = []
    index = 0
    for message in messages:
        if index >= max_sources:
            break
        body = compact(message.text, 700)
        if not body:
            continue
        index += 1
        index_to_message[index] = message
        posted = message_datetime_jst(message)[:10]
        source_lines.append(f"[{index}] (posted {posted}) {body}")

    if not source_lines:
        return [], index_to_message

    raw = openai_complete(settings, INVOICE_INSTRUCTIONS, "\n".join(source_lines)[:48000], timeout=150)
    parsed = parse_json_list(raw)

    records: list[dict] = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        n = obj.get("n")
        if not isinstance(n, int) or n not in index_to_message:
            continue
        due_raw = obj.get("due_date")
        due_date: date | None = None
        if isinstance(due_raw, str) and due_raw.strip():
            try:
                due_date = datetime.strptime(due_raw.strip()[:10], "%Y-%m-%d").date()
            except ValueError:
                due_date = None
        records.append(
            {
                "n": n,
                "vendor": (obj.get("vendor") or "").strip(),
                "amount": (obj.get("amount") or "").strip(),
                "due_date": due_date,
                "status": (obj.get("status") or "不明").strip(),
            }
        )

    # The model often splits one invoice into two rows — one carrying the amount
    # and a redundant one with an empty amount. For each (vendor, deadline):
    #   * if any row has a real amount, drop the empty-amount rows (noise);
    #   * keep rows with DISTINCT amounts (a vendor can have several invoices due
    #     the same day, e.g. 94,952円 and 182,754円 — those are real and kept);
    #   * if every row is empty, keep a single one.
    by_vendor_due: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_vendor_due[(r["vendor"], r["due_date"])].append(r)

    cleaned: list[dict] = []
    for rows in by_vendor_due.values():
        with_amount = [r for r in rows if r["amount"]]
        chosen = with_amount if with_amount else rows[:1]
        seen_amounts: set[str] = set()
        for r in chosen:
            if r["amount"] in seen_amounts:
                continue
            seen_amounts.add(r["amount"])
            cleaned.append(r)
    return cleaned, index_to_message


def thread_anchor(message: StoredMessage) -> str:
    """The ts to reply under: the thread parent if any, else the message itself."""
    return message.thread_ts or message.ts


def format_amount(amount: str) -> str:
    """Pretty-print an amount. Pure-digit values get thousands separators + 円;
    anything already formatted is kept exactly as the source wrote it."""
    raw = (amount or "").strip()
    if not raw:
        return ""
    digits = raw.replace(",", "").replace("，", "")
    if digits.isdigit():
        return f"{int(digits):,}円"
    return raw


# Known user-group handles -> subteam id. The bot token lacks usergroups:read,
# so handles cannot be resolved via the API; these ids were confirmed against the
# "【グループメンションについて】" guide message in #all-0_全体.
KNOWN_GROUPS = {
    "system-jin": "S0B7DKTNF4H",   # 内藤/るか/坪井 + JIN (4 people)
    "system-team": "S0AMJKD8EB1",  # 内藤/るか/坪井 (3 people)
}


def normalize_mention(token: str) -> str | None:
    """Turn a CLI mention token into Slack's mention syntax, or None if unknown.

    Accepts: already-formatted '<...>'; '@channel'/'@here'; a user id
    'U…'/'W…'; a user-group id 'S…'; or a known group handle such as
    'system-jin' (see KNOWN_GROUPS). An unknown bare handle returns None and the
    caller warns."""
    t = (token or "").strip()
    if not t:
        return None
    if t.startswith("<") and t.endswith(">"):
        return t
    bare = t.lstrip("@")
    if bare in ("channel", "everyone"):
        return "<!channel>"
    if bare == "here":
        return "<!here>"
    if bare.lower() in KNOWN_GROUPS:
        return f"<!subteam^{KNOWN_GROUPS[bare.lower()]}>"
    if re.fullmatch(r"[UW][A-Z0-9]{6,}", bare):
        return f"<@{bare}>"
    if re.fullmatch(r"S[A-Z0-9]{6,}", bare):
        return f"<!subteam^{bare}>"
    return None


def build_reminder_text(target: date, items: list[dict], test: bool = False,
                        mention_prefix: str = "") -> str:
    weekday = _JP_WEEKDAYS[target.weekday()]
    header = f"⏰ *支払期限リマインド｜明日 {target.month}/{target.day}({weekday}) が期限*"
    lines = []
    if test:
        lines.append("🧪 *【動作テスト】* これはリマインド機能の動作確認用です。確認後に削除してください。")
    if mention_prefix:
        lines.append(mention_prefix)
    lines.extend([header, "未入金の可能性がある請求です。ご確認をお願いします。"])
    for item in items:
        vendor = item["vendor"] or "（取引先不明）"
        amount = format_amount(item["amount"])
        amount = f"： {amount}" if amount else ""
        note = "" if item["status"] != "不明" else "（入金状況は未確認）"
        lines.append(f"• {vendor}{amount}{note}")
    lines.append("※すでに入金・処理済みでしたら、このリマインドは無視してください。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="On-demand invoice payment-due reminder.")
    parser.add_argument("--channel", default=INVOICE_CHANNEL_ID,
                        help=f"Invoice channel ID to scan (default {INVOICE_CHANNEL_ID})")
    parser.add_argument("--lookback-days", type=int, default=60,
                        help="How far back to scan the invoice channel for open invoices (default 60)")
    parser.add_argument("--due-in-days", type=int, default=1,
                        help="Remind for invoices due exactly this many days from today (default 1 = tomorrow)")
    parser.add_argument("--include-paid", action="store_true",
                        help="Also remind for invoices already marked 入金済み (default: skip them)")
    parser.add_argument("--post", action="store_true",
                        help="Actually post reminders into Slack threads (otherwise dry run / console only)")
    parser.add_argument("--test", action="store_true",
                        help="Prefix each reminder with a visible 【動作テスト】 banner (for verification posts)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only post to at most this many threads (0 = no limit). Useful for a single test post.")
    parser.add_argument("--mention", default="system-jin",
                        help="Comma-separated mention targets to prepend (default: system-jin = the 4-person "
                             "System-JIN group). Accepts known handles (system-jin/system-team), user ids (U…), "
                             "group ids (S…), @here, @channel. Pass --mention \"\" to send with no mention.")
    args = parser.parse_args()

    mention_prefix = ""
    if args.mention:
        resolved: list[str] = []
        for token in args.mention.split(","):
            token = token.strip()
            if not token:
                continue
            m = normalize_mention(token)
            if m:
                resolved.append(m)
            else:
                print(f"[警告] メンション '{token}' はIDに解決できないため無視します "
                      f"(<@U...> / <!subteam^S...> / @here / @channel のいずれかで指定してください)")
        mention_prefix = " ".join(resolved)

    settings = get_settings()
    storage = Storage(settings)
    storage.init_schema()

    workspace_id = most_populated_workspace_id(storage)
    if not workspace_id:
        print("対象ワークスペースのデータがありません。")
        return

    now = datetime.now(JST)
    target = (now + timedelta(days=args.due_in_days)).date()
    start_ts = str((now - timedelta(days=args.lookback_days)).timestamp())

    messages = storage.list_messages(
        workspace_id=workspace_id,
        channel_id=args.channel,
        search_scope="channel",
        limit=5000,
        oldest_ts=start_ts,
    )
    # Oldest first so the model reads invoices chronologically.
    messages.sort(key=lambda m: float(m.ts) if m.ts.replace(".", "").isdigit() else 0.0)

    print("=" * 60)
    print(f"請求書チャンネル {args.channel} を走査: 直近{args.lookback_days}日 / {len(messages)}件")
    print(f"対象期限: {target.isoformat()} (今日+{args.due_in_days}日)")

    if not messages:
        print("対象メッセージがありません。")
        return

    records, index_to_message = extract_invoices(settings, messages)

    # Show every extracted record with a real due date, so the values can be eyeballed.
    dated = [r for r in records if r["due_date"]]
    print(f"\n抽出された請求(期限あり) {len(dated)}件:")
    for r in sorted(dated, key=lambda r: r["due_date"]):
        print(f"  - {r['due_date'].isoformat()} | {r['status']:<4} | {r['vendor']} {r['amount']}")
    no_date = [r for r in records if not r["due_date"]]
    if no_date:
        print(f"  （期限の記載が無く対象外: {len(no_date)}件）")

    # Match: deadline == target, and not already paid (unless --include-paid).
    matches = [
        r for r in records
        if r["due_date"] == target and (args.include_paid or r["status"] != "入金済み")
    ]

    if not matches:
        print(f"\n→ {target.isoformat()} が期限の未入金請求はありません。リマインドなし。")
        print("=" * 60)
        return

    # Group matches by the Slack thread they should be posted into.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in matches:
        message = index_to_message[r["n"]]
        groups[(message.channel_id, thread_anchor(message))].append(r)

    group_items = list(groups.items())
    if args.limit and args.limit > 0:
        group_items = group_items[:args.limit]

    print(f"\n→ リマインド対象 {len(matches)}件 / スレッド {len(groups)}件"
          + (f"（うち {len(group_items)}件に投稿）" if args.limit else "") + ":")
    slack_client = SlackClient(settings) if args.post else None
    posted = 0
    for (channel_id, anchor), items in group_items:
        text = build_reminder_text(target, items, test=args.test, mention_prefix=mention_prefix)
        permalink = index_to_message[items[0]["n"]].permalink or "(no link)"
        print("\n--- スレッド " + f"{channel_id} @ {anchor}  {permalink}")
        print(text)
        if args.post and slack_client is not None:
            slack_client.post_message(channel=channel_id, text=text, thread_ts=anchor)
            posted += 1

    print("\n" + "=" * 60)
    if args.post:
        print(f"{posted}件のスレッドにリマインドを投稿しました。")
    else:
        print("(dry run) 実際に送るには --post を付けて再実行してください。")


if __name__ == "__main__":
    main()
