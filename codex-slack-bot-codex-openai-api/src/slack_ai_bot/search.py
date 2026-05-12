from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import logging
import math
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import Settings
from .openai_client import OpenAIClient
from .storage import Storage, StoredMessage

try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9), name="JST")
THREAD_MEMORY_HINTS = (
    "\u3053\u306e\u30b9\u30ec\u30c3\u30c9",
    "\u30b9\u30ec\u30c3\u30c9",
    "\u3055\u3063\u304d",  # sakki
    "\u3055\u304d\u307b\u3069",
    "\u3055\u3063\u304d\u306e",
    "\u3053\u306e\u4ef6",
    "\u3053\u306e\u5185\u5bb9",
    "\u4e0a\u306e",
    "\u4e0a\u8a18",
    "\u305d\u308c",
    "\u3053\u308c",
    "\u4eca\u306e",
    "\u4f1a\u8b70",
    "\u8981\u7d04",
    "\u307e\u3068\u3081",
    "\u6c7a\u307e\u3063\u305f",
    "\u30bf\u30b9\u30af",
    "\u672a\u5bfe\u5fdc",
    "\u8a73\u3057\u304f",
    "\u51fa\u3057\u3066",
    "\u3082\u3046\u4e00\u56de",
)
SLACK_LINK_RE = re.compile(r"/archives/([A-Z0-9]+)/p(\d{10})(\d{6})")
SLACK_THREAD_TS_RE = re.compile(r"[?&]thread_ts=(\d+\.\d+)")


@dataclass(frozen=True)
class SearchPlan:
    date_intent: bool
    date_reason: str
    start_date: str | None
    end_date: str | None
    oldest_ts: str | None
    latest_ts: str | None
    keywords: list[str]
    person_names: list[str]
    channel_names: list[str]


@dataclass(frozen=True)
class SlackMessageLink:
    channel_id: str
    ts: str
    thread_ts: str | None = None


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


def message_search_text(message: StoredMessage) -> str:
    parts = [
        message.channel_name or "",
        message.user_name or "",
        message.text,
    ]
    return "\n".join(part for part in parts if part)


def keyword_score(query: str, message: StoredMessage) -> float:
    query_tokens = keyword_tokens(query)
    if not query_tokens:
        return 0.0
    text = message_search_text(message).lower()
    text_tokens = keyword_tokens(text)
    overlap = query_tokens & text_tokens
    overlap_score = len(overlap) / max(len(query_tokens), 1)
    substring_hits = sum(1 for token in query_tokens if token in text)
    substring_score = substring_hits / max(len(query_tokens), 1)
    return max(overlap_score, substring_score)


def parse_slack_message_link(text: str) -> SlackMessageLink | None:
    match = SLACK_LINK_RE.search(text or "")
    if not match:
        return None
    channel_id, seconds, micros = match.groups()
    thread_match = SLACK_THREAD_TS_RE.search(text or "")
    return SlackMessageLink(
        channel_id=channel_id,
        ts=f"{seconds}.{micros}",
        thread_ts=thread_match.group(1) if thread_match else None,
    )


def message_key(message: StoredMessage) -> tuple[str, str, str]:
    return (message.workspace_id, message.channel_id, message.ts)


def message_sort_key(message: StoredMessage) -> float:
    try:
        return float(message.ts)
    except ValueError:
        return 0.0


def message_datetime_jst(message: StoredMessage) -> str:
    try:
        return datetime.fromtimestamp(float(message.ts), JST).strftime("%Y-%m-%d %H:%M:%S JST")
    except ValueError:
        return "unknown"


def thread_root_ts(message: StoredMessage) -> str:
    return message.thread_ts or message.ts


def build_search_query(question: str, plan: SearchPlan) -> str:
    parts = [question]
    if plan.keywords:
        parts.append("keywords: " + " ".join(plan.keywords))
    if plan.person_names:
        parts.append("people: " + " ".join(plan.person_names))
    if plan.channel_names:
        parts.append("channels: " + " ".join(plan.channel_names))
    if plan.start_date and plan.end_date:
        parts.append(f"date: {plan.start_date} to {plan.end_date}")
    return "\n".join(parts)


def search_terms(question: str, plan: SearchPlan) -> list[str]:
    terms: list[str] = []
    terms.extend(plan.keywords)
    terms.extend(plan.person_names)
    terms.extend(plan.channel_names)
    terms.extend(keyword_tokens(question))

    seen: set[str] = set()
    cleaned: list[str] = []
    for term in terms:
        value = term.strip().lower()
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def merge_messages(groups: list[list[StoredMessage]]) -> list[StoredMessage]:
    merged: list[StoredMessage] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for message in group:
            key = message_key(message)
            if key in seen:
                continue
            seen.add(key)
            merged.append(message)
    return merged


def string_contains_any(text: str | None, needles: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(needle.lower() in lowered for needle in needles if needle)


def metadata_score(plan: SearchPlan, message: StoredMessage) -> float:
    score = 0.0
    text = message_search_text(message)

    if plan.channel_names and string_contains_any(message.channel_name, plan.channel_names):
        score += 0.25
    if plan.person_names and (
        string_contains_any(message.user_name, plan.person_names)
        or string_contains_any(text, plan.person_names)
    ):
        score += 0.35
    if plan.keywords:
        score += min(keyword_score(" ".join(plan.keywords), message), 0.4)

    return min(score, 1.0)


def hybrid_score(
    query: str,
    message: StoredMessage,
    plan: SearchPlan,
    query_embedding: list[float] | None,
) -> float:
    embedding = 0.0
    if query_embedding and message.embedding:
        embedding = max(cosine_similarity(query_embedding, message.embedding), 0.0)

    keyword = keyword_score(query, message)
    metadata = metadata_score(plan, message)

    if query_embedding and message.embedding:
        return (embedding * 0.72) + (keyword * 0.18) + (metadata * 0.10)
    return (keyword * 0.75) + (metadata * 0.25)


def select_top_thread_hits(scored: list[tuple[float, StoredMessage]], limit: int) -> list[StoredMessage]:
    hits: list[StoredMessage] = []
    seen_threads: set[tuple[str, str, str]] = set()

    for _, message in scored:
        key = (message.workspace_id, message.channel_id, thread_root_ts(message))
        if key in seen_threads:
            continue
        seen_threads.add(key)
        hits.append(message)
        if len(hits) >= limit:
            break

    return hits


def contains_excluded_mention(message: StoredMessage, excluded_mention_ids: set[str] | None) -> bool:
    if not excluded_mention_ids:
        return False
    return any(f"<@{mention_id}>" in message.text for mention_id in excluded_mention_ids)


def exclude_mentioned_messages(
    messages: list[StoredMessage],
    excluded_mention_ids: set[str] | None,
) -> list[StoredMessage]:
    if not excluded_mention_ids:
        return messages
    return [message for message in messages if not contains_excluded_mention(message, excluded_mention_ids)]


def should_use_thread_memory(question: str, thread_ts: str | None, current_ts: str | None) -> bool:
    if not thread_ts or thread_ts == current_ts:
        return False
    lowered = question.lower()
    return any(hint in lowered for hint in THREAD_MEMORY_HINTS)


def thread_memory_messages(
    workspace_id: str,
    channel_id: str,
    thread_ts: str | None,
    current_ts: str | None,
    question: str,
    storage: Storage,
) -> list[StoredMessage]:
    if not should_use_thread_memory(question, thread_ts, current_ts):
        return []
    return storage.list_thread_messages(
        workspace_id=workspace_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
    )


def today_jst() -> str:
    return datetime.now(JST).date().isoformat()


def date_to_slack_ts(value: str) -> str:
    day = date.fromisoformat(value)
    at_midnight = datetime.combine(day, time.min, JST)
    return str(at_midnight.timestamp())


def safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_search_plan(raw_plan: dict[str, Any]) -> SearchPlan:
    date_intent = bool(raw_plan.get("date_intent"))
    start_date = raw_plan.get("start_date") if isinstance(raw_plan.get("start_date"), str) else None
    end_date = raw_plan.get("end_date") if isinstance(raw_plan.get("end_date"), str) else None
    oldest_ts = None
    latest_ts = None

    if date_intent and start_date and end_date:
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            days = (end - start).days
            if 0 < days <= 31:
                oldest_ts = date_to_slack_ts(start_date)
                latest_ts = date_to_slack_ts(end_date)
            else:
                date_intent = False
                start_date = None
                end_date = None
        except ValueError:
            date_intent = False
            start_date = None
            end_date = None

    return SearchPlan(
        date_intent=date_intent and bool(oldest_ts and latest_ts),
        date_reason=str(raw_plan.get("date_reason") or ""),
        start_date=start_date,
        end_date=end_date,
        oldest_ts=oldest_ts,
        latest_ts=latest_ts,
        keywords=safe_string_list(raw_plan.get("keywords")),
        person_names=safe_string_list(raw_plan.get("person_names")),
        channel_names=safe_string_list(raw_plan.get("channel_names")),
    )


def plan_search(question: str, openai_client: OpenAIClient) -> SearchPlan:
    try:
        return normalize_search_plan(openai_client.plan_search(question, today_jst()))
    except Exception:
        logging.exception("failed to create search plan")
        return normalize_search_plan({"date_intent": False})


def expand_context_messages(
    hits: list[StoredMessage],
    storage: Storage,
    settings: Settings,
    plan: SearchPlan,
    excluded_mention_ids: set[str] | None = None,
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
                oldest_ts=plan.oldest_ts if plan.date_intent else None,
                latest_ts=plan.latest_ts if plan.date_intent else None,
            )
        )

        if not group:
            group.append(hit)

        for message in sorted(group, key=message_sort_key):
            if contains_excluded_mention(message, excluded_mention_ids):
                continue
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
    thread_ts: str | None = None,
    current_ts: str | None = None,
    excluded_mention_ids: set[str] | None = None,
) -> list[StoredMessage]:
    linked_message = parse_slack_message_link(question)
    if linked_message:
        direct = storage.list_thread_messages(
            workspace_id=workspace_id,
            channel_id=linked_message.channel_id,
            thread_ts=linked_message.thread_ts or linked_message.ts,
        )
        direct = exclude_mentioned_messages(direct, excluded_mention_ids)
        if direct:
            return direct

    thread_memory = thread_memory_messages(
        workspace_id=workspace_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
        current_ts=current_ts,
        question=question,
        storage=storage,
    )
    if thread_memory:
        return exclude_mentioned_messages(thread_memory, excluded_mention_ids)

    plan = plan_search(question, openai_client)
    recent_candidates = storage.list_messages(
        workspace_id=workspace_id,
        channel_id=channel_id,
        search_scope=settings.search_scope,
        limit=settings.max_search_rows,
        oldest_ts=plan.oldest_ts if plan.date_intent else None,
        latest_ts=plan.latest_ts if plan.date_intent else None,
    )
    term_candidates = storage.list_messages_matching_terms(
        workspace_id=workspace_id,
        channel_id=channel_id,
        search_scope=settings.search_scope,
        terms=search_terms(question, plan),
        limit=settings.max_search_rows,
        oldest_ts=plan.oldest_ts if plan.date_intent else None,
        latest_ts=plan.latest_ts if plan.date_intent else None,
    )
    candidates = merge_messages([term_candidates, recent_candidates])
    candidates = exclude_mentioned_messages(candidates, excluded_mention_ids)
    if not candidates:
        return []

    search_query = build_search_query(question, plan)
    query_embedding: list[float] | None = None
    try:
        query_embedding = openai_client.create_embedding(search_query)
    except Exception:
        logging.exception("failed to create search embedding")

    scored: list[tuple[float, StoredMessage]] = []
    for message in candidates:
        score = hybrid_score(search_query, message, plan, query_embedding)
        if score > 0:
            scored.append((score, message))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = select_top_thread_hits(scored, settings.max_context_messages)
    elif plan.date_intent:
        hits = sorted(candidates, key=message_sort_key)[: settings.max_context_messages]
    else:
        hits = []

    return expand_context_messages(hits, storage, settings, plan, excluded_mention_ids=excluded_mention_ids)


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
            f"[{index}] {channel} {user} datetime_jst={message_datetime_jst(message)} "
            f"ts={message.ts} thread_root={root} source={message.source_type}\n"
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
