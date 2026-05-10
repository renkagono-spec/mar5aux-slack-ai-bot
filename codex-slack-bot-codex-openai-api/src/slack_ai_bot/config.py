from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_set(name: str) -> set[str]:
    value = os.getenv(name, "")
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str
    slack_signing_secret: str
    openai_api_key: str
    database_url: str | None
    sqlite_path: str
    openai_model: str
    openai_embedding_model: str
    search_scope: str
    max_search_rows: int
    max_context_messages: int
    embed_on_ingest: bool
    allowed_answer_channel_ids: set[str]
    slack_own_bot_id: str | None
    max_slack_event_age_seconds: int

    @property
    def missing_required_values(self) -> list[str]:
        missing = []
        if not self.slack_bot_token:
            missing.append("SLACK_BOT_TOKEN")
        if not self.slack_signing_secret:
            missing.append("SLACK_SIGNING_SECRET")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.database_url:
            missing.append("DATABASE_URL")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    search_scope = os.getenv("SEARCH_SCOPE", "workspace").strip().lower()
    if search_scope not in {"channel", "workspace"}:
        raise ValueError("SEARCH_SCOPE must be either 'channel' or 'workspace'")

    return Settings(
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        database_url=os.getenv("DATABASE_URL") or None,
        sqlite_path=os.getenv("SQLITE_PATH", "data/slack-ai.sqlite3"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        search_scope=search_scope,
        max_search_rows=env_int("MAX_SEARCH_ROWS", 5000),
        max_context_messages=env_int("MAX_CONTEXT_MESSAGES", 12),
        embed_on_ingest=env_bool("EMBED_ON_INGEST", True),
        allowed_answer_channel_ids=env_set("ALLOWED_ANSWER_CHANNEL_IDS"),
        slack_own_bot_id=os.getenv("SLACK_OWN_BOT_ID") or None,
        max_slack_event_age_seconds=env_int("MAX_SLACK_EVENT_AGE_SECONDS", 300),
    )
