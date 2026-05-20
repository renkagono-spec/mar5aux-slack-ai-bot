from __future__ import annotations

import json
import logging
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

    def plan_search(self, question: str, today_jst: str) -> dict[str, Any]:
        instructions = (
            "You are a careful Slack search planner. Return only compact JSON. "
            "Decide whether numbers like 4/27 are dates only from context. "
            "Treat 4/27 as a date when the question says things like 'on 4/27', "
            "'4/27 ni hanasareta', 'posts from 4/27', 'that day', 'yesterday', or asks what was discussed. "
            "Do not treat 4/27 as a date when it looks like a ratio, product number, size, count, or code. "
            "Use JST dates. If a year is omitted, use the year from today_jst. "
            "Always extract important Japanese nouns, person names, channel-like words, project names, and topic words into keywords. "
            "For questions about who/when/where, include the subject person and topic in keywords even when a date filter is used. "
            "JSON schema: "
            "{"
            "\"date_intent\": true|false, "
            "\"date_reason\": string, "
            "\"start_date\": \"YYYY-MM-DD\"|null, "
            "\"end_date\": \"YYYY-MM-DD\"|null, "
            "\"keywords\": [string], "
            "\"person_names\": [string], "
            "\"channel_names\": [string]"
            "}. "
            "end_date is exclusive. For a single day, end_date is the next day."
        )
        response = post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": self.settings.openai_model,
                "instructions": instructions,
                "input": f"today_jst={today_jst}\nquestion={question}",
                "temperature": 0,
            },
            headers=self.headers,
            timeout=60,
        )

        text = response.get("output_text") or ""
        if not text:
            chunks: list[str] = []
            for item in response.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        chunks.append(content["text"])
            text = "\n".join(chunks)

        try:
            first = text.find("{")
            last = text.rfind("}")
            if first >= 0 and last >= first:
                text = text[first : last + 1]
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logging.exception("failed to parse search plan")

        return {
            "date_intent": False,
            "date_reason": "planner_fallback",
            "start_date": None,
            "end_date": None,
            "keywords": [],
            "person_names": [],
            "channel_names": [],
        }

    def resolve_followup_question(self, question: str, thread_context: str, today_jst: str) -> dict[str, Any]:
        instructions = (
            "You rewrite a Slack follow-up question into a standalone Slack search question. "
            "Return only compact JSON. "
            "FIRST decide whether the current question CONTINUES the thread's topic or INTRODUCES a new one. "
            "A question CONTINUES the thread when it relies on the thread to be understood: it uses vague words "
            "such as 'soko', 'sore', 'kore', 'that', 'there', 'above', 'sakki'; it is elliptical (e.g. 'and the amount?', "
            "'is it written in email too?', 'please do it'); or it asks more about the same subject already discussed. "
            "For these, set uses_thread_context=true and rewrite into a standalone question that carries over the specific "
            "topic, dates, person names, project names, and channel names from the thread context. "
            "A question INTRODUCES a new topic when it names a concrete new entity, product, person, project, or subject "
            "that is NOT present in the thread context (for example asking about a different product or a different company). "
            "For these, set uses_thread_context=false and return the question UNCHANGED. "
            "Never attach the thread's channel name, topic, dates, or people to a question about an unrelated new subject. "
            "When unsure whether the question is related, prefer uses_thread_context=false and keep the question unchanged. "
            "Do not answer the question. Do not invent facts that are not in the thread context. "
            "Examples: "
            "Thread is about 'Tanaka Seni invoice'. Question 'is it written in email too?' -> "
            "uses_thread_context=true, standalone='Is the Tanaka Seni invoice also written about in any email?'. "
            "Thread is about 'Tanaka Seni invoice'. Question 'tell me the status of the Baleno emblem' -> "
            "uses_thread_context=false, standalone unchanged (Baleno emblem is a new unrelated subject; never merge it with the invoice topic). "
            "JSON schema: {"
            "\"uses_thread_context\": true|false, "
            "\"standalone_question\": string, "
            "\"reason\": string"
            "}. "
            "If the question is already standalone, return it unchanged with uses_thread_context=false."
        )
        response = post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": self.settings.openai_model,
                "instructions": instructions,
                "input": (
                    f"today_jst={today_jst}\n\n"
                    f"Current Slack question:\n{question}\n\n"
                    f"Thread context before the current question:\n{thread_context[:9000]}"
                ),
                "temperature": 0,
            },
            headers=self.headers,
            timeout=60,
        )

        text = response.get("output_text") or ""
        if not text:
            chunks: list[str] = []
            for item in response.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        chunks.append(content["text"])
            text = "\n".join(chunks)

        try:
            first = text.find("{")
            last = text.rfind("}")
            if first >= 0 and last >= first:
                text = text[first : last + 1]
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                standalone = str(parsed.get("standalone_question") or question).strip()
                return {
                    "uses_thread_context": bool(parsed.get("uses_thread_context")),
                    "standalone_question": standalone or question,
                    "reason": str(parsed.get("reason") or ""),
                }
        except json.JSONDecodeError:
            logging.exception("failed to parse follow-up resolution")

        return {
            "uses_thread_context": False,
            "standalone_question": question,
            "reason": "resolver_fallback",
        }

    def answer_question(self, question: str, context: str) -> str:
        instructions = (
            "Answer in Japanese. You are an internal Slack search assistant. "
            "Use only the supplied Context as evidence and keep the answer concise and specific. "
            "Start with the direct answer first. For questions asking when, where, who, or what, put the extracted value in the first sentence. "
            "The Context contains relevant search hits, same-thread replies, and nearby channel messages. "
            "When the user's question says 'sakki', 'this', 'that', 'above', or asks for a meeting summary inside a Slack thread, "
            "treat the supplied same-thread messages as short-term memory for that thread. "
            "Read thread_root, datetime_jst, timestamps, and neighboring messages before deciding the meaning. "
            "If a date-specific question includes later same-thread replies, distinguish the original day's discussion from later follow-ups. "
            "Never answer by describing the user's request itself. If the Context only contains bot requests or meta discussion, say that source content was not found. "
            "Do not invent facts. If the evidence is insufficient, say what is missing. "
            "Prefer concrete timestamps, dates, channel names, people, and meeting locations over vague summaries. "
            "If the answer depends on a later thread reply after the user's requested date, explain that timeline clearly. "
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
