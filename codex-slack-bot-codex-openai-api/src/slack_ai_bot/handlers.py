from __future__ import annotations

import logging
import re
from typing import Any

from .config import Settings
from .openai_client import OpenAIClient
from .search import format_context, search_messages, today_jst
from .slack_client import SlackClient
from .storage import Storage, StoredMessage


MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
CITATION_RE = re.compile(r"\[(\d{1,3})\]")
REFERENCE_LINKS_LABEL = "\u53c2\u7167\u30ea\u30f3\u30af"
FOLLOWUP_ONLY_HINTS = (
    "\u3088\u308d\u3057\u304f",
    "\u304a\u9858\u3044",
    "\u304a\u306d\u304c\u3044",
    "\u983c\u3080",
    "\u305f\u306e\u3080",
    "\u3084\u3063\u3066",
    "\u305d\u308c\u3067",
    "\u305d\u308c\u304a\u9858\u3044",
)
EXPLICIT_THREAD_REQUEST_HINTS = (
    "\u3053\u306e\u30b9\u30ec\u30c3\u30c9",
    "\u30b9\u30ec\u30c3\u30c9",
)
CONTEXTUAL_FOLLOWUP_HINTS = (
    "\u305d\u3053\u3067",
    "\u305d\u3053",
    "\u305d\u306e",
    "\u305d\u308c",
    "\u3053\u306e\u4ef6",
    "\u3053\u306e\u5185\u5bb9",
    "\u3055\u3063\u304d",
    "\u3055\u304d\u307b\u3069",
    "\u4e0a\u306e",
    "\u4e0a\u8a18",
    "\u524d\u306e",
    "\u3058\u3083\u3042",
)
SUBSTANTIVE_REQUEST_HINTS = (
    "?",
    "\uff1f",
    "4/",
    "\u6559\u3048\u3066",
    "\u63a2\u3057\u3066",
    "\u307e\u3068\u3081",
    "\u8981\u7d04",
    "\u6982\u8981",
    "\u5185\u5bb9",
    "\u8ab0",
    "\u4f55",
    "\u3044\u3064",
    "\u3069\u3053",
)


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


def compact_text(text: str) -> str:
    return re.sub(r"[\s\u3000\u3001\u3002,.!?\uff01\uff1f]+", "", text or "").lower()


def ts_as_float(ts: str | None) -> float:
    try:
        return float(ts or "0")
    except ValueError:
        return 0.0


def is_followup_only_question(question: str) -> bool:
    compact = compact_text(clean_question(question))
    if not compact:
        return False
    if any(hint in compact for hint in SUBSTANTIVE_REQUEST_HINTS):
        return False
    return len(compact) <= 24 and any(hint in compact for hint in FOLLOWUP_ONLY_HINTS)


def is_explicit_thread_request(question: str) -> bool:
    cleaned = clean_question(question).lower()
    return any(hint in cleaned for hint in EXPLICIT_THREAD_REQUEST_HINTS)


def is_contextual_followup_question(question: str) -> bool:
    if is_explicit_thread_request(question):
        return False
    cleaned = clean_question(question).lower()
    return any(hint in cleaned for hint in CONTEXTUAL_FOLLOWUP_HINTS)


def is_substantive_user_request(text: str) -> bool:
    cleaned = clean_question(text)
    if is_followup_only_question(cleaned):
        return False
    if any(hint in cleaned for hint in SUBSTANTIVE_REQUEST_HINTS):
        return True
    return len(compact_text(cleaned)) >= 14


def latest_previous_user_request(messages: list[StoredMessage], current_ts: str | None) -> str | None:
    current = ts_as_float(current_ts)
    for message in sorted(messages, key=lambda item: ts_as_float(item.ts), reverse=True):
        if current and ts_as_float(message.ts) >= current:
            continue
        if message.source_type == "bot_message" or not message.user_id:
            continue
        if not is_substantive_user_request(message.text):
            continue
        return clean_question(message.text)
    return None


def latest_previous_bot_answer(messages: list[StoredMessage], current_ts: str | None) -> str | None:
    current = ts_as_float(current_ts)
    for message in sorted(messages, key=lambda item: ts_as_float(item.ts), reverse=True):
        if current and ts_as_float(message.ts) >= current:
            continue
        if message.source_type != "bot_message":
            continue
        cleaned = message.text.strip()
        if cleaned:
            return cleaned[:3000]
    return None


def resolve_effective_question(
    question: str,
    current_thread_messages: list[StoredMessage],
    current_ts: str | None,
) -> tuple[str, bool]:
    if not current_thread_messages:
        return question, False

    previous_question = latest_previous_user_request(current_thread_messages, current_ts)
    previous_answer = latest_previous_bot_answer(current_thread_messages, current_ts)

    if is_contextual_followup_question(question) and (previous_question or previous_answer):
        return (
            "Previous user request in this Slack thread:\n"
            f"{previous_question or '(none)'}\n\n"
            "Previous assistant answer in this Slack thread:\n"
            f"{previous_answer or '(none)'}\n\n"
            "Current follow-up question:\n"
            f"{question}\n\n"
            "Resolve words like 'there', 'that', 'sore', or 'soko' from the previous thread context, "
            "then search the actual Slack source messages and answer the current follow-up directly.",
            True,
        )

    if not is_followup_only_question(question):
        return question, False
    if not previous_question:
        return question, False
    return previous_question, True


def should_resolve_with_ai(question: str, current_thread_messages: list[StoredMessage]) -> bool:
    return bool(
        current_thread_messages
        and (
            is_followup_only_question(question)
            or is_contextual_followup_question(question)
        )
    )


def format_thread_context_for_resolution(
    messages: list[StoredMessage],
    current_ts: str | None,
    max_messages: int = 12,
    max_chars: int = 9000,
) -> str:
    previous_messages = [
        message
        for message in sorted(messages, key=lambda item: ts_as_float(item.ts))
        if ts_as_float(message.ts) < ts_as_float(current_ts)
    ][-max_messages:]

    lines: list[str] = []
    total = 0
    for message in previous_messages:
        speaker = "assistant" if message.source_type == "bot_message" else (message.user_name or message.user_id or "unknown")
        body = message.text.strip()
        entry = f"{message.ts} {speaker}:\n{body}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)

    return "\n\n".join(lines)


def resolve_question_with_ai(
    question: str,
    current_thread_messages: list[StoredMessage],
    current_ts: str | None,
    openai_client: OpenAIClient,
) -> tuple[str, bool]:
    if not should_resolve_with_ai(question, current_thread_messages):
        return resolve_effective_question(question, current_thread_messages, current_ts)

    thread_context = format_thread_context_for_resolution(current_thread_messages, current_ts)
    if not thread_context:
        return resolve_effective_question(question, current_thread_messages, current_ts)

    try:
        resolution = openai_client.resolve_followup_question(question, thread_context, today_jst())
    except Exception:
        logging.exception("failed to resolve follow-up question with AI")
        return resolve_effective_question(question, current_thread_messages, current_ts)

    standalone = str(resolution.get("standalone_question") or question).strip()
    uses_thread_context = bool(resolution.get("uses_thread_context"))
    if uses_thread_context and standalone:
        return standalone, True
    return resolve_effective_question(question, current_thread_messages, current_ts)


def answer_question_text_for_thread_memory(question: str) -> str:
    return (
        f"{question}\n\n"
        "The source search returned no direct hits. Use the supplied Slack thread memory only when it directly answers the question. "
        "If it only contains the user's request and not the needed facts, say the source content was not found."
    )


def should_store_message(event: dict[str, Any], own_bot_id: str | None, include_own_bot: bool = False) -> bool:
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
    if own_bot_id and event.get("bot_id") == own_bot_id and not include_own_bot:
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
    include_own_bot: bool = False,
    embed: bool = True,
) -> StoredMessage | None:
    own_bot_id = slack_client.own_bot_id()
    if not should_store_message(event, own_bot_id, include_own_bot=include_own_bot):
        return None

    text = event_text(event)
    if not text:
        return None

    channel_id = event["channel"]
    ts = event["ts"]
    embedding = None
    if settings.embed_on_ingest and embed:
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


def live_thread_messages_by_ts(
    payload: dict[str, Any],
    channel_id: str,
    thread_ts: str,
    slack_client: SlackClient,
    openai_client: OpenAIClient,
    settings: Settings,
) -> list[StoredMessage]:
    if not channel_id or not thread_ts:
        return []

    cursor = None
    messages: list[StoredMessage] = []
    while True:
        response = slack_client.conversation_replies(channel=channel_id, ts=thread_ts, cursor=cursor)
        for reply_event in response.get("messages") or []:
            reply_event.setdefault("channel", channel_id)
            message = message_from_event(
                payload,
                reply_event,
                slack_client,
                openai_client,
                settings,
                include_own_bot=True,
                embed=False,
            )
            if message:
                messages.append(message)

        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return messages


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
        current_thread_messages: list[StoredMessage] = []
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
            try:
                current_thread_messages = live_thread_messages_by_ts(
                    payload,
                    channel_id,
                    event["thread_ts"],
                    slack_client,
                    openai_client,
                    settings,
                )
            except Exception:
                logging.exception("failed to fetch live thread memory")
                current_thread_messages = storage.list_thread_messages(
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    thread_ts=event["thread_ts"],
                )

        effective_question, inherited_question = resolve_question_with_ai(
            question,
            current_thread_messages,
            event.get("ts"),
            openai_client,
        )
        excluded_mention_ids = {mention_id for mention_id in [slack_client.own_user_id()] if mention_id}

        matches = search_messages(
            question=effective_question,
            workspace_id=workspace_id,
            channel_id=channel_id,
            settings=settings,
            storage=storage,
            openai_client=openai_client,
            thread_ts=event.get("thread_ts") if not inherited_question else None,
            current_ts=event.get("ts") if not inherited_question else None,
            excluded_mention_ids=excluded_mention_ids,
        )

        if not matches and inherited_question and current_thread_messages:
            thread_memory = [
                message
                for message in current_thread_messages
                if ts_as_float(message.ts) < ts_as_float(event.get("ts"))
            ]
            thread_context_messages = thread_memory[-settings.max_context_messages :]
            context = format_context(thread_context_messages, max_chars=settings.max_context_chars)
            answer = openai_client.answer_question(
                answer_question_text_for_thread_memory(effective_question),
                context,
            )
            answer_with_links = answer.rstrip() + format_evidence_links(answer, thread_context_messages)
            slack_client.post_message(channel=channel_id, thread_ts=thread_ts, text=answer_with_links[:39000])
            return

        if not matches:
            slack_client.post_message(
                channel=channel_id,
                thread_ts=thread_ts,
                text="No relevant Slack information was found yet. Invite the bot to target channels and backfill history first.",
            )
            return

        matches = refresh_threads_for_matches(payload, matches, storage, slack_client, openai_client, settings)
        if excluded_mention_ids:
            matches = [
                message
                for message in matches
                if not any(f"<@{mention_id}>" in message.text for mention_id in excluded_mention_ids)
            ]
        if not matches:
            slack_client.post_message(
                channel=channel_id,
                thread_ts=thread_ts,
                text="\u691c\u7d22\u3067\u304d\u308b\u53c2\u7167\u5143\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002bot\u3078\u306e\u4f9d\u983c\u6587\u3067\u306f\u306a\u304f\u3001\u5b9f\u969b\u306eSlack\u6295\u7a3f\u304cDB\u306b\u5165\u3063\u3066\u3044\u308b\u304b\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            )
            return
        context = format_context(matches, max_chars=settings.max_context_chars)
        answer_question_text = effective_question
        if inherited_question:
            answer_question_text = (
                f"{effective_question}\n\n"
                "The latest Slack reply was only a short follow-up. "
                "Answer the request above directly. Do not summarize the follow-up or the request itself."
            )
        answer = openai_client.answer_question(answer_question_text, context)
        answer_with_links = answer.rstrip() + format_evidence_links(answer, matches)
        slack_client.post_message(channel=channel_id, thread_ts=thread_ts, text=answer_with_links[:39000])
    except Exception:
        logging.exception("failed to answer app mention")
        slack_client.post_message(
            channel=channel_id,
            thread_ts=thread_ts,
            text="An error occurred while answering. Please check the server logs and API key settings.",
        )
