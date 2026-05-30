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
            "Use ONLY what the supplied Context messages literally say. Be concise and specific.\n\n"
            "STEP 1 — IDENTIFY THE SUBJECT of the question. The subject is the specific thing being asked about: "
            "the product, project, item, person, deal, or topic. Write it down internally before doing anything else.\n\n"
            "STEP 2 — TOPICAL-MATCH GATE (must pass before any answer is given):\n"
            "Scan each Context message. A message is usable ONLY if its own text explicitly mentions the subject, "
            "or it is a same-thread reply whose parent message explicitly mentions the subject. "
            "Same-time / same-word / same-verb matches are NOT enough.\n"
            "Examples:\n"
            "- Question: '事務所の家具はいつ見に行く予定？' (subject = 事務所の家具). "
            "  A message saying only '5/31 見に行く' in a thread about ポロシャツの量産 does NOT mention 家具 or 事務所, "
            "  so it is NOT usable. The correct answer is 「該当する情報が見つかりませんでした」.\n"
            "- Question: '田中センイの請求はいくら？' (subject = 田中センイの請求). "
            "  A message saying '田中センイ様の請求 ¥314,536' explicitly mentions the subject, so it IS usable.\n"
            "If NO Context message passes the gate, answer exactly: "
            "「該当する情報が見つかりませんでした」 and stop. Do not pull dates, names, or details from messages that fail the gate.\n\n"
            "STEP 3 — ANSWER from the messages that passed the gate:\n"
            "- Start with the direct answer. For when/where/who/what questions, put the extracted value in the first sentence.\n"
            "- Use only what the messages literally state. Never infer or fabricate dates, times, schedules, deadlines, "
            "amounts, prices, payment status, or decisions that are not explicitly written.\n"
            "- If a specific sub-detail is not in the gated messages, say it is not stated rather than guessing.\n"
            "- Prefer concrete timestamps, dates, channel names, people, and meeting locations over vague summaries.\n"
            "- If the answer depends on a later same-thread reply, explain that timeline clearly.\n"
            "- Treat same-thread replies as short-term memory only for ambiguous references like 'sakki', 'this', 'that', 'above', "
            "or thread-summary requests. Read thread_root, datetime_jst, and neighboring messages first.\n"
            "- Never answer by describing the user's request itself. If the Context only contains bot requests or meta discussion, "
            "say that source content was not found.\n\n"
            "STEP 4 — CITATIONS:\n"
            "- Attach source markers like [1] only to the gated messages you actually relied on. "
            "Do not cite messages that failed the gate.\n"
            "- Do not paste Slack URLs in the body; use only [n] markers. The server will append links only for the markers you used.\n"
            "- If there are unresolved, undecided, or needs-confirmation items, separate them at the end.\n\n"
            "HARD RULE — when the gate would fail (the subject 家具/事務所/その他特定の主題 is not literally mentioned "
            "in any candidate message), you MUST output exactly:「該当する情報が見つかりませんでした」 followed by "
            "one short line saying what the question's subject was, and stop. Do not list dates from unrelated messages. "
            "Do not write 'predict', 'probably', 'likely', or any softened guess in this case."
        )
        user_input = f"Question:\n{question}\n\nContext:\n{context}"

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "instructions": instructions,
            "input": user_input,
            "temperature": 0,
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
