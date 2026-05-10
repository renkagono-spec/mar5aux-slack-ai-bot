from __future__ import annotations

import logging
import re
from typing import Any

from .config import Settings
from .openai_client import OpenAIClient
from .search import format_context, search_messages
from .slack_client import SlackClient
from .storage import Storage, StoredMessage


MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
CITATION_RE = re.compile(r"\[(\d{1,3})\]")
REFERENCE_LINKS_LABEL = "\u53c2\u7167\u30ea\u30f3\u30af"


def workspace_id_from_payload(payload: dict[str, Any], event: dict[str, Any]) -> str:
    return (
        payload.get("team_id")
        or event.get("team")
        or payload.get("authorizations", [{}])[0].get("team_id")
        or "unknown-workspace"
    )


def clean_question(text: str) -> str:
    cleaned = MENTION_RE.sub("", text or "").strip()
    return cleaned or "Summarize the relevant information in this thread or channel."


def should_store_message(event: dict[str, Any], own_bot_id: str | None) -> bool:
    subtype = event.get("subtype")
    if subtype in {
        "channel_join",
        "channel_leave",
        "channel_archive",
        "channel_unarchive",
        "group_join",
        "group_leave",
        "message_deleted",
        "message_changed",
    }:
        return False
    if own_bot_id and event.get("bot_id") == own_bot_id:
        return False
    return bool((event.get("text") or "").strip() or event.get("files"))


def event_text(event: dict[str, Any]) -> str:
    text = (event.get("text") or "").strip()
    files = event.get("files") or []
    if not files:
        return text

    file_bits = []
    for file in files:
        title = file.get("title") or file.get("name") or "untitled"
        mimetype = file.get("mimetype") or file.get("filetype") or "unknown"
        file_bits.append(f"[file: {title} ({mimetype})]")
    return "\n".join([part for part in [text, *file_bits] if part])


def message_from_event(
    payload: dict[str, Any],
    event: dict[str, Any],
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> StoredMessage | None:
    own_bot_id = slack_client.own_bot_id()
    if not should_store_message(event, own_bot_id):
        return None

    text = event_text(event)
    if not text:
        return None

    channel_id = event["channel"]
    ts = event["ts"]
    embedding = None
    if settings.embed_on_ingest:
        try:
            embedding = openai_client.create_embedding(text)
        except Exception:
            logging.exception("failed to create embedding")

    subtype = event.get("subtype")
    if event.get("files") or subtype == "file_share":
        source_type = "slack_file"
    elif event.get("bot_id"):
        source_type = "bot_message"
    else:
        source_type = "slack_message"

    return StoredMessage(
        workspace_id=workspace_id_from_payload(payload, event),
        channel_id=channel_id,
        channel_name=slack_client.channel_name(channel_id),
        user_id=event.get("user"),
        user_name=slack_client.user_name(event.get("user")),
        ts=ts,
        thread_ts=event.get("thread_ts"),
        text=text,
        permalink=slack_client.get_permalink(channel_id, ts),
        source_type=source_type,
        embedding=embedding,
    )


def store_thread_replies(
    payload: dict[str, Any],
    event: dict[str, Any],
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> int:
    channel_id = event.get("channel") or event.get("message", {}).get("channel")
    parent = event.get("message") if event.get("subtype") == "message_replied" else event
    parent_ts = parent.get("thread_ts") or parent.get("ts")
    if not channel_id or not parent_ts:
        return 0

    return store_thread_by_ts(payload, channel_id, parent_ts, storage, slack_client, openai_client, settings)


def store_thread_by_ts(
    payload: dict[str, Any],
    channel_id: str,
    thread_ts: str,
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> int:
    if not channel_id or not thread_ts:
        return 0

    cursor = None
    stored = 0
    while True:
        response = slack_client.conversation_replies(channel=channel_id, ts=thread_ts, cursor=cursor)
        for reply_event in response.get("messages") or []:
            reply_event.setdefault("channel", channel_id)
            message = message_from_event(payload, reply_event, slack_client, openai_client, settings)
            if message:
                storage.upsert_message(message, raw=reply_event)
                stored += 1

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return stored


def refresh_threads_for_matches(
    payload: dict[str, Any],
    matches: list[StoredMessage],
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> list[StoredMessage]:
    roots: list[tuple[str, str, str]] = []
    seen_roots: set[tuple[str, str, str]] = set()

    for message in matches[: settings.max_context_messages]:
        root_ts = message.thread_ts or message.ts
        key = (message.workspace_id, message.channel_id, root_ts)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(key)
        try:
            store_thread_by_ts(payload, message.channel_id, root_ts, storage, slack_client, openai_client, settings)
        except Exception:
            logging.exception("failed to refresh matched thread")

    refreshed: list[StoredMessage] = []
    seen_messages: set[tuple[str, str, str]] = set()
    for workspace_id, channel_id, root_ts in roots:
        for message in storage.list_thread_messages(workspace_id, channel_id, root_ts):
            key = (message.workspace_id, message.channel_id, message.ts)
            if key in seen_messages:
                continue
            seen_messages.add(key)
            refreshed.append(message)

    return refreshed or matches


def cited_message_indexes(answer: str, message_count: int) -> list[int]:
    indexes: list[int] = []
    seen: set[int] = set()
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if index < 1 or index > message_count or index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    return indexes


def format_evidence_links(answer: str, messages: list[StoredMessage], max_links: int = 4) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    for index in cited_message_indexes(answer, len(messages)):
        message = messages[index - 1]
        if not message.permalink or message.permalink in seen:
            continue
        seen.add(message.permalink)
        channel = f"#{message.channel_name}" if message.channel_name else message.channel_id
        label = f"{channel} {message.ts}"
        lines.append(f"{len(lines) + 1}. <{message.permalink}|{label}>")
        if len(lines) >= max_links:
            break

    if not lines:
        return ""
    return f"\n\n{REFERENCE_LINKS_LABEL}:\n" + "\n".join(lines)


def handle_message_event(
    payload: dict[str, Any],
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> None:
    event = payload.get("event", {})
    subtype = event.get("subtype")
    workspace_id = workspace_id_from_payload(payload, event)

    if subtype == "message_deleted":
        previous = event.get("previous_message") or {}
        ts = previous.get("ts") or event.get("deleted_ts")
        if ts and event.get("channel"):
            storage.mark_deleted(workspace_id, event["channel"], ts)
        return

    if subtype == "message_replied":
        try:
            store_thread_replies(payload, event, storage, slack_client, openai_client, settings)
        except Exception:
            logging.exception("failed to store thread replies")
        return

    if subtype == "message_changed":
        changed = event.get("message") or {}
        if event.get("channel") and "channel" not in changed:
            changed["channel"] = event["channel"]
        message = message_from_event(payload, changed, slack_client, openai_client, settings)
    else:
        message = message_from_event(payload, event, slack_client, openai_client, settings)

    if message:
        storage.upsert_message(message, raw=event)


def handle_app_mention(
    payload: dict[str, Any],
    storage: Storage,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> None:
    event = payload.get("event", {})
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts") or event.get("ts")

    if settings.allowed_answer_channel_ids and channel_id not in settings.allowed_answer_channel_ids:
        slack_client.post_message(
            channel=channel_id,
            thread_ts=thread_ts,
            text="This bot is configured to answer only in approved channels. Please ask the admin to add this channel.",
        )
        return

    question = clean_question(event.get("text", ""))
    workspace_id = workspace_id_from_payload(payload, event)
    slack_client.add_reaction(channel_id, event["ts"], "eyes")

    try:
        if event.get("thread_ts"):
            store_thread_by_ts(
                payload,
                channel_id,
                event["thread_ts"],
                storage,
                slack_client,
                openai_client,
                settings,
            )

        matches = search_messages(
            question=question,
            workspace_id=workspace_id,
            channel_id=channel_id,
            settings=settings,
            storage=storage,
            openai_client=openai_client,
            thread_ts=event.get("thread_ts"),
            current_ts=event.get("ts"),
        )

        if not matches:
            slack_client.post_message(
                channel=channel_id,
                thread_ts=thread_ts,
                text="No relevant Slack information was found yet. Invite the bot to target channels and backfill history first.",
            )
            return

        matches = refresh_threads_for_matches(payload, matches, storage, slack_client, openai_client, settings)
        context = format_context(matches, max_chars=settings.max_context_chars)
        answer = openai_client.answer_question(question, context)
        answer_with_links = answer.rstrip() + format_evidence_links(answer, matches)
        slack_client.post_message(channel=channel_id, thread_ts=thread_ts, text=answer_with_links[:39000])
    except Exception:
        logging.exception("failed to answer app mention")
        slack_client.post_message(
            channel=channel_id,
            thread_ts=thread_ts,
            text="An error occurred while answering. Please check the server logs and API key settings.",
        )
