from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slack_ai_bot.config import get_settings
from slack_ai_bot.handlers import live_thread_messages_by_ts, resolve_question_with_ai
from slack_ai_bot.openai_client import OpenAIClient
from slack_ai_bot.search import format_context, message_datetime_jst, plan_search, search_messages
from slack_ai_bot.slack_client import SlackClient
from slack_ai_bot.storage import Storage, StoredMessage


def ts_as_float(value: str | None) -> float:
    try:
        return float(value or "0")
    except ValueError:
        return 0.0


def own_user_id(settings) -> str | None:
    if not settings.slack_bot_token:
        return None
    try:
        return SlackClient(settings).own_user_id()
    except Exception:
        return None


def most_populated_workspace_id(storage: Storage) -> str:
    with storage.connect() as conn:
        if storage.backend == "postgres":
            row = conn.execute(
                """
                SELECT workspace_id
                FROM messages
                WHERE is_deleted = FALSE
                GROUP BY workspace_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT workspace_id
                FROM messages
                WHERE is_deleted = 0
                GROUP BY workspace_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """
            ).fetchone()
    if not row:
        raise SystemExit("No messages are indexed yet.")
    return dict(row)["workspace_id"]


def most_populated_channel_id(storage: Storage, workspace_id: str) -> str:
    with storage.connect() as conn:
        if storage.backend == "postgres":
            row = conn.execute(
                """
                SELECT channel_id
                FROM messages
                WHERE workspace_id = %s AND is_deleted = FALSE
                GROUP BY channel_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT channel_id
                FROM messages
                WHERE workspace_id = ? AND is_deleted = 0
                GROUP BY channel_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
    if not row:
        raise SystemExit(f"No messages are indexed for workspace_id={workspace_id}.")
    return dict(row)["channel_id"]


def print_plan(question: str, openai_client: OpenAIClient) -> None:
    plan = plan_search(question, openai_client)
    print("search_plan:")
    print(f"  date_intent={plan.date_intent}")
    print(f"  date_reason={plan.date_reason}")
    print(f"  start_date={plan.start_date}")
    print(f"  end_date={plan.end_date}")
    print(f"  keywords={plan.keywords}")
    print(f"  person_names={plan.person_names}")
    print(f"  channel_names={plan.channel_names}")


def print_matches(matches: list[StoredMessage], limit: int) -> None:
    print(f"matches={len(matches)}")
    for index, message in enumerate(matches[:limit], start=1):
        channel = f"#{message.channel_name}" if message.channel_name else message.channel_id
        user = f"@{message.user_name}" if message.user_name else (message.user_id or "unknown")
        body = " ".join(message.text.split())
        if len(body) > 180:
            body = body[:180] + "..."
        print(
            f"[{index}] {channel} {user} {message_datetime_jst(message)} "
            f"ts={message.ts} thread_ts={message.thread_ts or '-'}"
        )
        print(f"    {body}")
        if message.permalink:
            print(f"    {message.permalink}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug how the Slack AI bot resolves a question, searches DB context, and optionally answers."
    )
    parser.add_argument("question", nargs="+", help="Question text to test")
    parser.add_argument("--workspace-id", help="Workspace ID. Defaults to the most indexed workspace.")
    parser.add_argument("--channel-id", help="Question channel ID. Defaults to the most indexed channel.")
    parser.add_argument("--thread-ts", help="Slack thread root ts for follow-up/memory debugging")
    parser.add_argument("--current-ts", help="Current message ts. Used to ignore messages after this point in thread memory.")
    parser.add_argument("--live-thread", action="store_true", help="Fetch the thread from Slack before resolving the question")
    parser.add_argument("--answer", action="store_true", help="Also call OpenAI to generate the final answer")
    parser.add_argument("--show-context", action="store_true", help="Print the full formatted context sent to OpenAI")
    parser.add_argument("--match-limit", type=int, default=20, help="Maximum matched messages to print")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    settings = get_settings()
    storage = Storage(settings)
    storage.init_schema()
    openai_client = OpenAIClient(settings)

    workspace_id = args.workspace_id or most_populated_workspace_id(storage)
    channel_id = args.channel_id or most_populated_channel_id(storage, workspace_id)

    thread_messages: list[StoredMessage] = []
    if args.thread_ts:
        if args.live_thread:
            slack_client = SlackClient(settings)
            payload = {"team_id": workspace_id}
            thread_messages = live_thread_messages_by_ts(
                payload,
                channel_id,
                args.thread_ts,
                slack_client,
                openai_client,
                settings,
            )
        else:
            thread_messages = storage.list_thread_messages(workspace_id, channel_id, args.thread_ts)

    current_ts = args.current_ts
    if not current_ts and thread_messages:
        current_ts = str(max(ts_as_float(message.ts) for message in thread_messages) + 0.000001)

    effective_question, inherited = resolve_question_with_ai(
        question,
        thread_messages,
        current_ts,
        openai_client,
    )

    print(f"workspace_id={workspace_id}")
    print(f"channel_id={channel_id}")
    print(f"input_question={question}")
    print(f"effective_question={effective_question}")
    print(f"used_thread_memory={inherited}")
    print("")
    print_plan(effective_question, openai_client)
    print("")

    excluded_mention_ids = {own_user_id(settings)}
    matches = search_messages(
        question=effective_question,
        workspace_id=workspace_id,
        channel_id=channel_id,
        settings=settings,
        storage=storage,
        openai_client=openai_client,
        thread_ts=args.thread_ts if not inherited else None,
        current_ts=current_ts if not inherited else None,
        excluded_mention_ids={item for item in excluded_mention_ids if item},
    )
    print_matches(matches, args.match_limit)

    context = format_context(matches, max_chars=settings.max_context_chars)
    if args.show_context:
        print("")
        print("context:")
        print(context)

    if args.answer:
        print("")
        print("answer:")
        print(openai_client.answer_question(effective_question, context))


if __name__ == "__main__":
    main()
