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
            "あなたは社内Slack情報を検索して答える日本語の業務アシスタントです。"
            "与えられたContextだけを根拠に、短く具体的に回答してください。"
            "Contextには検索でヒットした投稿、その同一スレッド、前後の近い投稿が含まれます。"
            "単独投稿だけで判断せず、thread_root、時刻、前後関係を見て文脈を読み取ってください。"
            "推測で断定せず、根拠が足りない場合は不足していると明示してください。"
            "重要な主張には [1] のような参照番号を付け、Slack permalink がある場合は必ず併記してください。"
            "リンクを省略しないでください。リンクが多い場合でも、主要な根拠リンクは残してください。"
            "未対応・未決定・要確認の話題は最後に分けてください。"
        )
        user_input = f"質問:\n{question}\n\nContext:\n{context}"

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
        return "\n".join(chunks).strip() or "回答を生成できませんでした。"
