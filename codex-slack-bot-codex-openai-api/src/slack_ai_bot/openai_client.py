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
            "For katakana foreign words, loanwords, and brand/company names, ALSO add their likely Latin-script spellings to "
            "keywords so the logs (often written in English) are matched. Examples: エスライド -> add 'S-RIDE', 'sride', 'S.RIDE'; "
            "マネーフォワード -> add 'MoneyForward', 'moneyforward'; ショッピファイ -> add 'Shopify'. Keep BOTH the katakana and the "
            "Latin spellings. Do not invent unrelated words. "
            "Keywords must be SPECIFIC entities, names, brands, or topic nouns. Do NOT put generic filler words in keywords "
            "(e.g. 内容, こと, もの, どこ, いつ, ある, 記載, 教えて, ついて, 関して). "
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
                "model": self.settings.openai_fast_model,
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
                "model": self.settings.openai_fast_model,
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
            "Answer in Japanese. You are an internal Slack search assistant that answers ONLY from the supplied "
            "Context messages. Be concise and specific.\n\n"
            "GROUNDING (most important — never break these):\n"
            "- Use ONLY facts literally written in the Context. Never invent or guess dates, times, deadlines, amounts, "
            "prices, payment status, decisions, or next actions that are not explicitly written.\n"
            "- Do NOT combine separate messages into a claim that none of them makes on its own. Every statement must be "
            "supported by a specific message that actually says it.\n"
            "- Attach a source marker like [1] to each statement, using only numbers that appear in the Context. Never paste "
            "raw Slack URLs; the server adds links for the [n] markers you use.\n\n"
            "WHICH MESSAGES TO USE:\n"
            "- Use the messages that are about the SUBJECT of the question (the product, project, person, deal, company, "
            "channel, or topic asked about). Differences in wording are fine — judge by meaning, not exact string match.\n"
            "- A romanized/Latin spelling or obvious spelling variant of the same entity is the SAME subject "
            "(e.g. エスライド = S-RIDE = sride; マネーフォワード = MoneyForward). Channel names help too: a question about the WEB / "
            "サイト is well served by #mar5aux-web messages; a billing question by #請求書 messages.\n"
            "- Ignore messages that are clearly about a DIFFERENT topic — do not borrow their dates or numbers. Also ignore "
            "messages that are only requests to this bot or meta-discussion.\n\n"
            "WHO DID WHAT (mail direction):\n"
            "- A mail entry's header may carry mail=受信/送信 with from= and to=. mail=受信 is INBOUND: the action is the "
            "sender's (from=); the mailbox owner (to=) only received it. mail=送信 is OUTBOUND: sent BY the mailbox owner.\n"
            "- Answer by who actually performed the thing asked about. If asked what a person DID / handled / sent / replied / "
            "decided, count only what that person authored — their own Slack messages and their 送信 mail — and do NOT present "
            "受信 mail merely addressed to them as their action (that is the sender's action). If asked what a person RECEIVED "
            "or was contacted about, use their 受信 mail. A channel named after a person is that person's mailbox, not proof "
            "they acted.\n\n"
            "ANSWER STYLE:\n"
            "- Lead with the direct answer; put the key value (when / where / who / what / how much) in the first sentence.\n"
            "- For progress or status questions, give a short chronological list of the concrete updates found, each with its "
            "date and [n].\n"
            "- If a specific sub-detail is not stated, say so plainly instead of guessing.\n"
            "- Put any unresolved / undecided / needs-confirmation items at the end.\n\n"
            "NOT FOUND: answer 「該当する情報が見つかりませんでした」 ONLY when none of the Context messages actually address the "
            "question. Do NOT answer not-found merely because the wording differs from the question — if relevant messages "
            "exist, answer from them."
        )
        user_input = f"Question:\n{question}\n\nContext:\n{context}"

        payload: dict[str, Any] = {
            "model": self.settings.openai_answer_model,
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
