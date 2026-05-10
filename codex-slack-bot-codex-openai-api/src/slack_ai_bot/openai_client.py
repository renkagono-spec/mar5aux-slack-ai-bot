from __future__ import annotations

from typing import Any

from .config import Settings
from .http_json import post_json


class OpenAIClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.openai_api_key}"}

    def create_embedding(self, text: str) -> list[float] | None:
        cleaned = text.strip()
        if not cleaned or not self.settings.openai_api_key:
            return None

        response = post_json(
            "https://api.openai.com/v1/embeddings",
            {
                "model": self.settings.openai_embedding_model,
                "input": cleaned[:12000],
            },
            headers=self.headers,
            timeout=60,
        )
        data = response.get("data") or []
        if not data:
            return None
        return data[0].get("embedding")

    def answer_question(self, question: str, context: str) -> str:
        instructions = (
            "Answer in Japanese. You are an internal Slack search assistant. "
            "Use only the supplied Context as evidence and keep the answer concise and specific. "
            "The Context contains relevant search hits, same-thread replies, and nearby channel messages. "
            "Read thread_root, timestamps, and neighboring messages before deciding the meaning. "
            "Do not invent facts. If the evidence is insufficient, say what is missing. "
            "Attach source markers like [1] to important claims. "
            "Do not paste Slack URLs in the answer body; use only source markers such as [1]. "
            "The server will append links only for the source markers you actually used. "
            "If there are unresolved, undecided, or needs-confirmation items, separate them at the end."
        )
        user_input = f"Question:\n{question}\n\nContext:\n{context}"

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "instructions": instructions,
            "input": user_input,
        }

        response = post_json(
            "https://api.openai.com/v1/responses",
            payload,
            headers=self.headers,
            timeout=90,
        )

        if response.get("output_text"):
            return response["output_text"].strip()

        chunks: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(content["text"])
        return "\n".join(chunks).strip() or "Could not generate an answer."
