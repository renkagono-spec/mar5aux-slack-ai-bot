from __future__ import annotations

import logging
import math
import re

from .config import Settings
from .openai_client import OpenAIClient
from .storage import Storage, StoredMessage


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def keyword_tokens(text: str) -> set[str]:
    lowered = text.lower()
    split_pattern = "[\\s\\u3001\\u3002,.!?\\uff01\\uff1f:\\uff1a/()\\[\\]{}<>\\u300c\\u300d\\u300e\\u300f]+"
    return {token for token in re.split(split_pattern, lowered) if len(token) >= 2}


def keyword_score(query: str, message: StoredMessage) -> float:
    query_tokens = keyword_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = keyword_tokens(message.text)
    overlap = query_tokens & text_tokens
    return len(overlap) / max(len(query_tokens), 1)


def message_key(message: StoredMessage) -> tuple[str, str, str]:
    return (message.workspace_id, message.channel_id, message.ts)


def message_sort_key(message: StoredMessage) -> float:
    try:
        return float(message.ts)
    except ValueError:
        return 0.0


def thread_root_ts(message: StoredMessage) -> str:
    return message.thread_ts or message.ts


def expand_context_messages(
    hits: list[StoredMessage],
    storage: Storage,
    settings: Settings,
) -> list[StoredMessage]:
    expanded: list[StoredMessage] = []
    seen: set[tuple[str, str, str]] = set()

    for hit in hits:
        group: list[StoredMessage] = []
        group.extend(
            storage.list_thread_messages(
                workspace_id=hit.workspace_id,
                channel_id=hit.channel_id,
                thread_ts=thread_root_ts(hit),
            )
        )
        group.extend(
            storage.list_neighbor_messages(
                workspace_id=hit.workspace_id,
                channel_id=hit.channel_id,
                ts=hit.ts,
                before=settings.context_neighbor_messages,
                after=settings.context_neighbor_messages,
            )
        )

        if not group:
            group.append(hit)

        for message in sorted(group, key=message_sort_key):
            key = message_key(message)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(message)

    return expanded


def search_messages(
    question: str,
    workspace_id: str,
    channel_id: str,
    settings: Settings,
    storage: Storage,
    openai_client: OpenAIClient,
) -> list[StoredMessage]:
    candidates = storage.list_messages(
        workspace_id=workspace_id,
        channel_id=channel_id,
        search_scope=settings.search_scope,
        limit=settings.max_search_rows,
    )

    query_embedding: list[float] | None = None
    try:
        query_embedding = openai_client.create_embedding(question)
    except Exception:
        logging.exception("failed to create search embedding")

    scored: list[tuple[float, StoredMessage]] = []

    if query_embedding:
        for message in candidates:
            if message.embedding:
                score = cosine_similarity(query_embedding, message.embedding)
                if score > 0:
                    scored.append((score, message))

    if not scored:
        for message in candidates:
            score = keyword_score(question, message)
            if score > 0:
                scored.append((score, message))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [message for _, message in scored[: settings.max_context_messages]]
    return expand_context_messages(hits, storage, settings)


def format_context(messages: list[StoredMessage], max_chars: int = 26000) -> str:
    lines: list[str] = [
        "The context below includes relevant Slack hits plus same-thread replies and nearby channel messages.",
        "Messages are grouped around the most relevant hits and ordered chronologically within each group.",
    ]
    total_chars = sum(len(line) for line in lines)

    for index, message in enumerate(messages, start=1):
        channel = f"#{message.channel_name}" if message.channel_name else message.channel_id
        user = f"@{message.user_name}" if message.user_name else (message.user_id or "unknown")
        permalink = message.permalink or "no permalink"
        root = thread_root_ts(message)
        body = message.text.strip()
        if len(body) > 2500:
            body = body[:2500] + "\n[message truncated]"

        entry = (
            f"[{index}] {channel} {user} ts={message.ts} thread_root={root} source={message.source_type}\n"
            f"permalink: {permalink}\n"
            f"{body}"
        )
        entry_size = len(entry) + 2
        if total_chars + entry_size > max_chars:
            lines.append("[context truncated because it exceeded the configured size limit]")
            break

        lines.append(entry)
        total_chars += entry_size

    return "\n\n".join(lines)
