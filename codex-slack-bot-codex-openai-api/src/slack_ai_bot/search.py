from __future__ import annotations

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
    return {token for token in re.split(r"[\s、。,.!?！？:：/()\[\]{}<>「」『』]+", lowered) if len(token) >= 2}


def keyword_score(query: str, message: StoredMessage) -> float:
    query_tokens = keyword_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = keyword_tokens(message.text)
    overlap = query_tokens & text_tokens
    return len(overlap) / max(len(query_tokens), 1)


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

    query_embedding = openai_client.create_embedding(question)
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
    return [message for _, message in scored[: settings.max_context_messages]]


def format_context(messages: list[StoredMessage]) -> str:
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        channel = f"#{message.channel_name}" if message.channel_name else message.channel_id
        user = f"@{message.user_name}" if message.user_name else (message.user_id or "unknown")
        permalink = message.permalink or "permalinkなし"
        lines.append(
            f"[{index}] {channel} {user} ts={message.ts} source={message.source_type}\n"
            f"permalink: {permalink}\n"
            f"{message.text.strip()}"
        )
    return "\n\n".join(lines)
