"""Agentic answer path.

The old bot did ONE retrieval (top-12) then ONE answer, so it could not count,
enumerate, or dig — unlike a capable assistant that runs several lookups, checks
the data, and only then answers. This module gives the model tools and lets it
loop: decide -> call a tool -> read the result -> maybe call another -> answer.
That generalizes across question types instead of needing a patch per type.
"""

from __future__ import annotations

import json
import logging
import re

from .aggregate import query_emails
from .config import Settings
from .openai_client import OpenAIClient
from .search import message_datetime_jst, search_messages, today_jst
from .storage import Storage, StoredMessage

MAX_STEPS = 4


def _compact(text: str, limit: int) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed[:limit] + ("…" if len(collapsed) > limit else "")

_INSTRUCTIONS = (
    "あなたは社内Slack/メールのアシスタントです。ツールを使って自分で調べ、確かめてから日本語で答えます。\n"
    "毎ターン、次のいずれかのJSONを *1つだけ* 返してください（前後に文章を付けない）:\n"
    '- 検索: {"action":"search","query":"検索語"}\n'
    '- メール集計: {"action":"query_emails","person":"人名/ハンドル","direction":"sent|received|any",'
    '"counterpart":"相手の会社やドメイン(任意)","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD","mode":"count|list"}\n'
    '- 最終回答: {"action":"final","answer":"日本語の回答"}\n\n'
    "ツールの説明:\n"
    "- search: 意味検索。関連メッセージを[n]付きで返す。『何を/なぜ/どうなってる/経緯』など内容系に使う。"
    "ただし上位数件しか見えないので『件数・網羅』には使わない。\n"
    "- query_emails: 転送メールを重複除去してDB全体から正確に集計/列挙する。"
    "『何件/合計/一覧/どこ宛て/誰から/それぞれの内容』は必ずこれを使う。"
    "person は調べたい社内の人（未指定可）、counterpart は相手先で絞る時に使う。"
    "end_date は含まないので、その月末までなら翌月1日を入れる。mode=list で各メールの中身も出る。\n\n"
    "進め方:\n"
    "- 件数・列挙の質問は search で数えようとせず query_emails を使う。\n"
    "- 内容を問われたら mode=list。足りなければ別のツールをもう一度呼ぶ。\n"
    "- 事実には search 結果の [n] を付ける。推測で数や日付を作らない。\n"
    "- 【最重要】query_emails が返した件数・内訳の数字は *一字一句そのまま* 使う。"
    "合算・四捨五入・省略・並べ替えをしない（例: 内訳の行を1つでも落とさない）。\n"
    "- 相手先で絞る時、counterpart はメールのドメインやアドレスの一部（例: tanakaseni, kawashima, resourceful）"
    "や表示名の姓（例: 田中）で指定すると当たりやすい。日本語の会社名しか分からない時は、"
    "まず counterpart 無しで direction 指定の内訳(by 相手)を取り、その中の該当ドメインの件数を読み取る。\n"
    "- 十分な材料が集まったら final で簡潔かつ具体的に答える。"
)


def _parse_action(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(cleaned[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def answer_with_agent(
    question: str,
    *,
    storage: Storage,
    openai_client: OpenAIClient,
    settings: Settings,
    workspace_id: str,
    channel_id: str,
    asker_id: str | None,
    asker_name: str | None,
    excluded_mention_ids: set[str] | None,
) -> tuple[str, list[StoredMessage]]:
    """Run the tool loop. Returns (answer_text, sources) for evidence links."""
    sources: list[StoredMessage] = []

    def tool_search(query: str) -> str:
        matches = search_messages(
            question=query,
            workspace_id=workspace_id,
            channel_id=channel_id,
            settings=settings,
            storage=storage,
            openai_client=openai_client,
            excluded_mention_ids=excluded_mention_ids,
            asker_id=asker_id,
            asker_name=asker_name,
        )
        if not matches:
            return "該当なし"
        out: list[str] = []
        for msg in matches[: settings.max_context_messages]:
            sources.append(msg)
            n = len(sources)
            body = _compact(msg.text, 300)
            out.append(f"[{n}] #{msg.channel_name} {message_datetime_jst(msg)[:16]}: {body}")
        return "\n".join(out)

    def tool_query_emails(args: dict) -> str:
        try:
            return query_emails(
                storage,
                person=str(args.get("person", "") or ""),
                direction=str(args.get("direction", "any") or "any"),
                counterpart=str(args.get("counterpart", "") or ""),
                start_date=args.get("start_date") or None,
                end_date=args.get("end_date") or None,
                mode=str(args.get("mode", "count") or "count"),
                asker_name=asker_name,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("query_emails tool failed")
            return f"(query_emailsエラー: {exc})"

    context_head = (
        f"今日は {today_jst()} (JST)。質問者(asker)= {asker_name or '不明'}。"
        "「私/自分」は質問者を指す。\n\n"
        f"ユーザーの質問: {question}\n"
    )
    transcript = context_head
    last_answer = ""
    last_email_output: str | None = None  # deterministic email count/list to return verbatim
    last_tool = ""

    for step in range(MAX_STEPS):
        force_final = step == MAX_STEPS - 1
        prompt = transcript + (
            "\nこれ以上ツールは使えません。これまでの観測だけで {\"action\":\"final\",...} を返してください。"
            if force_final else "\n次のアクションをJSONで1つ返してください。"
        )
        try:
            raw = openai_client.complete_text(_INSTRUCTIONS, prompt, timeout=90)
        except Exception:
            logging.exception("agent step failed")
            break
        action = _parse_action(raw)
        if not action:
            transcript += f"\n\n(注意: 直前の出力がJSONとして解釈できませんでした。JSONだけ返してください)"
            continue
        kind = action.get("action")
        if kind == "final":
            # Email counts/lists must be exact. mini composes prose faithfully but
            # miscounts when it aggregates a list itself, so when the last step was
            # an email query, return that deterministic output verbatim.
            if last_tool == "email" and last_email_output:
                return last_email_output, sources
            return str(action.get("answer") or "").strip(), sources
        if kind == "search":
            obs = tool_search(str(action.get("query", "") or question))
            last_tool = "search"
        elif kind == "query_emails":
            obs = tool_query_emails(action)
            last_email_output = obs
            last_tool = "email"
        else:
            obs = f"(未知のaction: {kind})"
        transcript += f"\n\n実行: {json.dumps(action, ensure_ascii=False)}\n観測:\n{obs[:2800]}"

    # ran out of steps without a final -> ask once more for a plain answer
    try:
        raw = openai_client.complete_text(
            _INSTRUCTIONS,
            transcript + "\n\nこれまでの観測をもとに、日本語で最終回答だけを書いてください（JSON不要）。",
            timeout=90,
        )
        last_answer = _parse_action(raw).get("answer", "") if _parse_action(raw) else raw
    except Exception:
        logging.exception("agent final failed")
    return (last_answer or "うまく回答をまとめられませんでした。"), sources
