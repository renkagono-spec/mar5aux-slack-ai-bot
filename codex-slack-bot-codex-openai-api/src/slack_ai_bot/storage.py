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
    ) -> list[StoredMessage]:
        channel_filter = search_scope == "channel" and channel_id
        with self.connect() as conn:
            if self.backend == "postgres":
                if channel_filter:
                    rows = conn.execute(
                        """
                        SELECT * FROM messages
                        WHERE workspace_id = %s AND channel_id = %s AND is_deleted = FALSE
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (workspace_id, channel_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM messages
                        WHERE workspace_id = %s AND is_deleted = FALSE
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (workspace_id, limit),
                    ).fetchall()
            else:
                if channel_filter:
                    rows = conn.execute(
                        """
                        SELECT * FROM messages
                        WHERE workspace_id = ? AND channel_id = ? AND is_deleted = 0
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (workspace_id, channel_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM messages
                        WHERE workspace_id = ? AND is_deleted = 0
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (workspace_id, limit),
                    ).fetchall()
        return [self._row_to_message(row) for row in rows]

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
