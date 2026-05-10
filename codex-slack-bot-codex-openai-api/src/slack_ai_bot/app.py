from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from .config import get_settings
from .handlers import handle_app_mention, handle_message_event
from .openai_client import OpenAIClient
from .slack_client import SlackClient
from .storage import Storage


logging.basicConfig(level=logging.INFO)

settings = get_settings()
storage = Storage(settings)
slack_client = SlackClient(settings)
openai_client = OpenAIClient(settings)


def verify_slack_signature(body: bytes, timestamp: str | None, signature: str | None) -> None:
    if not settings.slack_signing_secret:
        raise HTTPException(status_code=500, detail="SLACK_SIGNING_SECRET is not configured")
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="missing Slack signature")

    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid Slack timestamp") from exc

    if abs(time.time() - request_time) > settings.max_slack_event_age_seconds:
        raise HTTPException(status_code=401, detail="stale Slack request")

    base = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode("utf-8"),
        base,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="invalid Slack signature")


def create_app() -> FastAPI:
    api = FastAPI(title="Slack AI Search Bot", version="0.1.0")

    @api.on_event("startup")
    def startup() -> None:
        storage.init_schema()
        missing = settings.missing_required_values
        if missing:
            logging.warning("missing production settings: %s", ", ".join(missing))

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        try:
            storage.health_check()
        except Exception as exc:
            logging.exception("database health check failed")
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        return {
            "ok": True,
            "storage": storage.backend,
            "db": "ok",
            "search_scope": settings.search_scope,
            "features": [
                "thread_replies",
                "neighbor_context",
                "cited_evidence_links",
                "ai_search_planning",
                "thread_memory",
                "live_thread_refresh",
            ],
            "missing": settings.missing_required_values,
        }

    @api.post("/slack/events")
    async def slack_events(
        request: Request,
        background_tasks: BackgroundTasks,
        x_slack_request_timestamp: str | None = Header(default=None),
        x_slack_signature: str | None = Header(default=None),
    ) -> dict[str, Any]:
        body = await request.body()
        verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature)

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON") from exc

        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}

        if payload.get("type") != "event_callback":
            return {"ok": True}

        event_id = payload.get("event_id", "")
        if event_id and not storage.record_event(event_id):
            return {"ok": True, "duplicate": True}

        event = payload.get("event") or {}
        event_type = event.get("type")

        if event_type == "app_mention":
            background_tasks.add_task(
                handle_app_mention,
                payload,
                storage,
                slack_client,
                openai_client,
                settings,
            )
        elif event_type == "message":
            background_tasks.add_task(
                handle_message_event,
                payload,
                storage,
                slack_client,
                openai_client,
                settings,
            )

        return {"ok": True}

    return api


app = create_app()
