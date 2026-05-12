from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .config import Settings


@dataclass
class StoredMessage:
    workspace_id: str
    channel_id: str
    channel_name: str | None
    user_id: str | None
    user_name: str | None
    ts: str
    thread_ts: str | None
    text: str
    permalink: str | None
    source_type: str
    embedding: list[float] | None


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = "postgres" if settings.database_url else "sqlite"

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.backend == "postgres":
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(self.settings.database_url, row_factory=dict_row) as conn:
                yield conn
        else:
            db_path = Path(self.settings.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            if self.backend == "postgres":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS slack_events (
                        event_id TEXT PRIMARY KEY,
                        received_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGSERIAL PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        channel_id TEXT NOT NULL,
                        channel_name TEXT,
                        user_id TEXT,
                        user_name TEXT,
                        ts TEXT NOT NULL,
                        thread_ts TEXT,
                        text TEXT NOT NULL,
                        permalink TEXT,
                        source_type TEXT NOT NULL,
                        embedding_json TEXT,
                        raw_json JSONB,
                        is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(workspace_id, channel_id, ts)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_workspace_updated ON messages(workspace_id, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_updated ON messages(workspace_id, channel_id, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_thread_ts ON messages(workspace_id, channel_id, thread_ts, ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages(workspace_id, channel_id, ts)"
                )
            else:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS slack_events (
                        event_id TEXT PRIMARY KEY,
                        received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        channel_id TEXT NOT NULL,
                        channel_name TEXT,
                        user_id TEXT,
                        user_name TEXT,
                        ts TEXT NOT NULL,
                        thread_ts TEXT,
                        text TEXT NOT NULL,
                        permalink TEXT,
                        source_type TEXT NOT NULL,
                        embedding_json TEXT,
                        raw_json TEXT,
                        is_deleted INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(workspace_id, channel_id, ts)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_workspace_updated ON messages(workspace_id, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_updated ON messages(workspace_id, channel_id, updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_thread_ts ON messages(workspace_id, channel_id, thread_ts, ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages(workspace_id, channel_id, ts)"
                )

    def health_check(self) -> None:
        with self.connect() as conn:
            conn.execute("SELECT 1").fetchone()

    def record_event(self, event_id: str) -> bool:
        if not event_id:
            return True

        with self.connect() as conn:
            if self.backend == "postgres":
                row = conn.execute(
                    "INSERT INTO slack_events(event_id) VALUES (%s) ON CONFLICT DO NOTHING RETURNING event_id",
                    (event_id,),
                ).fetchone()
                return row is not None

            try:
                conn.execute("INSERT INTO slack_events(event_id) VALUES (?)", (event_id,))
                return True
            except sqlite3.IntegrityError:
                return False

    def upsert_message(self, message: StoredMessage, raw: dict[str, Any] | None = None) -> None:
        embedding_json = json.dumps(message.embedding) if message.embedding else None
        raw_json = json.dumps(raw or {}, ensure_ascii=False)
        values = (
            message.workspace_id,
            message.channel_id,
            message.channel_name,
            message.user_id,
            message.user_name,
            message.ts,
            message.thread_ts,
            message.text,
            message.permalink,
            message.source_type,
            embedding_json,
            raw_json,
        )

        with self.connect() as conn:
            if self.backend == "postgres":
                conn.execute(
                    """
                    INSERT INTO messages (
                        workspace_id, channel_id, channel_name, user_id, user_name, ts, thread_ts,
                        text, permalink, source_type, embedding_json, raw_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(workspace_id, channel_id, ts)
                    DO UPDATE SET
                        channel_name = EXCLUDED.channel_name,
                        user_id = EXCLUDED.user_id,
                        user_name = EXCLUDED.user_name,
                        thread_ts = EXCLUDED.thread_ts,
                        text = EXCLUDED.text,
                        permalink = EXCLUDED.permalink,
                        source_type = EXCLUDED.source_type,
                        embedding_json = COALESCE(EXCLUDED.embedding_json, messages.embedding_json),
                        raw_json = EXCLUDED.raw_json,
                        is_deleted = FALSE,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO messages (
                        workspace_id, channel_id, channel_name, user_id, user_name, ts, thread_ts,
                        text, permalink, source_type, embedding_json, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id, channel_id, ts)
                    DO UPDATE SET
                        channel_name = excluded.channel_name,
                        user_id = excluded.user_id,
                        user_name = excluded.user_name,
                        thread_ts = excluded.thread_ts,
                        text = excluded.text,
                        permalink = excluded.permalink,
                        source_type = excluded.source_type,
                        embedding_json = COALESCE(excluded.embedding_json, messages.embedding_json),
                        raw_json = excluded.raw_json,
                        is_deleted = 0,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values,
                )

    def mark_deleted(self, workspace_id: str, channel_id: str, ts: str) -> None:
        with self.connect() as conn:
            if self.backend == "postgres":
                conn.execute(
                    """
                    UPDATE messages
                    SET is_deleted = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE workspace_id = %s AND channel_id = %s AND ts = %s
                    """,
                    (workspace_id, channel_id, ts),
                )
            else:
                conn.execute(
                    """
                    UPDATE messages
                    SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE workspace_id = ? AND channel_id = ? AND ts = ?
                    """,
                    (workspace_id, channel_id, ts),
                )

    def list_messages(
        self,
        workspace_id: str,
        channel_id: str | None,
        search_scope: str,
        limit: int,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
    ) -> list[StoredMessage]:
        channel_filter = search_scope == "channel" and channel_id
        with self.connect() as conn:
            if self.backend == "postgres":
                conditions = ["workspace_id = %s", "is_deleted = FALSE"]
                params: list[Any] = [workspace_id]
                if channel_filter:
                    conditions.append("channel_id = %s")
                    params.append(channel_id)
                if oldest_ts:
                    conditions.append("ts::double precision >= %s::double precision")
                    params.append(oldest_ts)
                if latest_ts:
                    conditions.append("ts::double precision < %s::double precision")
                    params.append(latest_ts)
                params.append(limit)
                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE {" AND ".join(conditions)}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                ).fetchall()
            else:
                conditions = ["workspace_id = ?", "is_deleted = 0"]
                params = [workspace_id]
                if channel_filter:
                    conditions.append("channel_id = ?")
                    params.append(channel_id)
                if oldest_ts:
                    conditions.append("CAST(ts AS REAL) >= CAST(? AS REAL)")
                    params.append(oldest_ts)
                if latest_ts:
                    conditions.append("CAST(ts AS REAL) < CAST(? AS REAL)")
                    params.append(latest_ts)
                params.append(limit)
                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE {" AND ".join(conditions)}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def list_messages_matching_terms(
        self,
        workspace_id: str,
        channel_id: str | None,
        search_scope: str,
        terms: list[str],
        limit: int,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
    ) -> list[StoredMessage]:
        cleaned_terms = [term.strip().lower() for term in terms if len(term.strip()) >= 2]
        if not cleaned_terms:
            return []

        channel_filter = search_scope == "channel" and channel_id
        with self.connect() as conn:
            if self.backend == "postgres":
                conditions = ["workspace_id = %s", "is_deleted = FALSE"]
                params: list[Any] = [workspace_id]
                if channel_filter:
                    conditions.append("channel_id = %s")
                    params.append(channel_id)
                if oldest_ts:
                    conditions.append("ts::double precision >= %s::double precision")
                    params.append(oldest_ts)
                if latest_ts:
                    conditions.append("ts::double precision < %s::double precision")
                    params.append(latest_ts)

                searchable = "LOWER(COALESCE(text, '') || ' ' || COALESCE(channel_name, '') || ' ' || COALESCE(user_name, ''))"
                term_clauses = []
                for term in cleaned_terms[:12]:
                    term_clauses.append(f"{searchable} LIKE %s")
                    params.append(f"%{term}%")
                conditions.append("(" + " OR ".join(term_clauses) + ")")
                params.append(limit)

                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE {" AND ".join(conditions)}
                    ORDER BY ts::double precision DESC
                    LIMIT %s
                    """,
                    tuple(params),
                ).fetchall()
            else:
                conditions = ["workspace_id = ?", "is_deleted = 0"]
                params = [workspace_id]
                if channel_filter:
                    conditions.append("channel_id = ?")
                    params.append(channel_id)
                if oldest_ts:
                    conditions.append("CAST(ts AS REAL) >= CAST(? AS REAL)")
                    params.append(oldest_ts)
                if latest_ts:
                    conditions.append("CAST(ts AS REAL) < CAST(? AS REAL)")
                    params.append(latest_ts)

                searchable = "LOWER(COALESCE(text, '') || ' ' || COALESCE(channel_name, '') || ' ' || COALESCE(user_name, ''))"
                term_clauses = []
                for term in cleaned_terms[:12]:
                    term_clauses.append(f"{searchable} LIKE ?")
                    params.append(f"%{term}%")
                conditions.append("(" + " OR ".join(term_clauses) + ")")
                params.append(limit)

                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE {" AND ".join(conditions)}
                    ORDER BY CAST(ts AS REAL) DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def list_thread_messages(
        self,
        workspace_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> list[StoredMessage]:
        with self.connect() as conn:
            if self.backend == "postgres":
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE workspace_id = %s
                      AND channel_id = %s
                      AND is_deleted = FALSE
                      AND (ts = %s OR thread_ts = %s)
                    ORDER BY ts::double precision ASC
                    """,
                    (workspace_id, channel_id, thread_ts, thread_ts),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE workspace_id = ?
                      AND channel_id = ?
                      AND is_deleted = 0
                      AND (ts = ? OR thread_ts = ?)
                    ORDER BY CAST(ts AS REAL) ASC
                    """,
                    (workspace_id, channel_id, thread_ts, thread_ts),
                ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def list_neighbor_messages(
        self,
        workspace_id: str,
        channel_id: str,
        ts: str,
        before: int,
        after: int,
        oldest_ts: str | None = None,
        latest_ts: str | None = None,
    ) -> list[StoredMessage]:
        older: list[Any] = []
        newer: list[Any] = []

        with self.connect() as conn:
            if self.backend == "postgres":
                if before > 0:
                    oldest_clause = "AND ts::double precision >= %s::double precision" if oldest_ts else ""
                    latest_clause = "AND ts::double precision < %s::double precision" if latest_ts else ""
                    params: list[Any] = [workspace_id, channel_id, ts]
                    if oldest_ts:
                        params.append(oldest_ts)
                    if latest_ts:
                        params.append(latest_ts)
                    params.append(before)
                    older = conn.execute(
                        f"""
                        SELECT * FROM messages
                        WHERE workspace_id = %s
                          AND channel_id = %s
                          AND is_deleted = FALSE
                          AND (thread_ts IS NULL OR thread_ts = ts)
                          AND ts::double precision < %s::double precision
                          {oldest_clause}
                          {latest_clause}
                        ORDER BY ts::double precision DESC
                        LIMIT %s
                        """,
                        tuple(params),
                    ).fetchall()
                if after > 0:
                    oldest_clause = "AND ts::double precision >= %s::double precision" if oldest_ts else ""
                    latest_clause = "AND ts::double precision < %s::double precision" if latest_ts else ""
                    params = [workspace_id, channel_id, ts]
                    if oldest_ts:
                        params.append(oldest_ts)
                    if latest_ts:
                        params.append(latest_ts)
                    params.append(after)
                    newer = conn.execute(
                        f"""
                        SELECT * FROM messages
                        WHERE workspace_id = %s
                          AND channel_id = %s
                          AND is_deleted = FALSE
                          AND (thread_ts IS NULL OR thread_ts = ts)
                          AND ts::double precision > %s::double precision
                          {oldest_clause}
                          {latest_clause}
                        ORDER BY ts::double precision ASC
                        LIMIT %s
                        """,
                        tuple(params),
                    ).fetchall()
            else:
                if before > 0:
                    oldest_clause = "AND CAST(ts AS REAL) >= CAST(? AS REAL)" if oldest_ts else ""
                    latest_clause = "AND CAST(ts AS REAL) < CAST(? AS REAL)" if latest_ts else ""
                    params = [workspace_id, channel_id, ts]
                    if oldest_ts:
                        params.append(oldest_ts)
                    if latest_ts:
                        params.append(latest_ts)
                    params.append(before)
                    older = conn.execute(
                        f"""
                        SELECT * FROM messages
                        WHERE workspace_id = ?
                          AND channel_id = ?
                          AND is_deleted = 0
                          AND (thread_ts IS NULL OR thread_ts = ts)
                          AND CAST(ts AS REAL) < CAST(? AS REAL)
                          {oldest_clause}
                          {latest_clause}
                        ORDER BY CAST(ts AS REAL) DESC
                        LIMIT ?
                        """,
                        tuple(params),
                    ).fetchall()
                if after > 0:
                    oldest_clause = "AND CAST(ts AS REAL) >= CAST(? AS REAL)" if oldest_ts else ""
                    latest_clause = "AND CAST(ts AS REAL) < CAST(? AS REAL)" if latest_ts else ""
                    params = [workspace_id, channel_id, ts]
                    if oldest_ts:
                        params.append(oldest_ts)
                    if latest_ts:
                        params.append(latest_ts)
                    params.append(after)
                    newer = conn.execute(
                        f"""
                        SELECT * FROM messages
                        WHERE workspace_id = ?
                          AND channel_id = ?
                          AND is_deleted = 0
                          AND (thread_ts IS NULL OR thread_ts = ts)
                          AND CAST(ts AS REAL) > CAST(? AS REAL)
                          {oldest_clause}
                          {latest_clause}
                        ORDER BY CAST(ts AS REAL) ASC
                        LIMIT ?
                        """,
                        tuple(params),
                    ).fetchall()

        rows = list(reversed(older)) + list(newer)
        return [self._row_to_message(row) for row in rows]

    def channel_message_stats(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if self.backend == "postgres":
                conditions = ["is_deleted = FALSE"]
                params: list[Any] = []
                if workspace_id:
                    conditions.append("workspace_id = %s")
                    params.append(workspace_id)
                rows = conn.execute(
                    f"""
                    SELECT
                        workspace_id,
                        channel_id,
                        COALESCE(channel_name, channel_id) AS channel_name,
                        COUNT(*) AS message_count,
                        MIN(ts::double precision) AS oldest_ts,
                        MAX(ts::double precision) AS latest_ts,
                        COUNT(*) FILTER (WHERE thread_ts IS NOT NULL AND thread_ts <> ts) AS reply_count,
                        COUNT(*) FILTER (WHERE embedding_json IS NULL) AS missing_embedding_count
                    FROM messages
                    WHERE {" AND ".join(conditions)}
                    GROUP BY workspace_id, channel_id, channel_name
                    ORDER BY message_count DESC
                    """,
                    tuple(params),
                ).fetchall()
            else:
                conditions = ["is_deleted = 0"]
                params = []
                if workspace_id:
                    conditions.append("workspace_id = ?")
                    params.append(workspace_id)
                rows = conn.execute(
                    f"""
                    SELECT
                        workspace_id,
                        channel_id,
                        COALESCE(channel_name, channel_id) AS channel_name,
                        COUNT(*) AS message_count,
                        MIN(CAST(ts AS REAL)) AS oldest_ts,
                        MAX(CAST(ts AS REAL)) AS latest_ts,
                        SUM(CASE WHEN thread_ts IS NOT NULL AND thread_ts <> ts THEN 1 ELSE 0 END) AS reply_count,
                        SUM(CASE WHEN embedding_json IS NULL THEN 1 ELSE 0 END) AS missing_embedding_count
                    FROM messages
                    WHERE {" AND ".join(conditions)}
                    GROUP BY workspace_id, channel_id, channel_name
                    ORDER BY message_count DESC
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def _row_to_message(self, row: Any) -> StoredMessage:
        value = dict(row)
        embedding = None
        if value.get("embedding_json"):
            try:
                embedding = json.loads(value["embedding_json"])
            except json.JSONDecodeError:
                embedding = None
        return StoredMessage(
            workspace_id=value["workspace_id"],
            channel_id=value["channel_id"],
            channel_name=value.get("channel_name"),
            user_id=value.get("user_id"),
            user_name=value.get("user_name"),
            ts=value["ts"],
            thread_ts=value.get("thread_ts"),
            text=value["text"],
            permalink=value.get("permalink"),
            source_type=value["source_type"],
            embedding=embedding,
        )
