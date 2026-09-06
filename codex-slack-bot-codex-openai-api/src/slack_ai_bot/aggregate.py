"""Aggregate / counting answers.

Top-K retrieval can find relevant messages but cannot COUNT — it only ever sees
~12 messages, so "how many emails did X send" is answered from a tiny sample and
comes out wrong. This module answers count/enumerate questions about email by
querying the whole DB, deduping the forwarder's mirror/burst copies, and counting
deterministically. The email address is the identity, so no person registry is
needed for the common case.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from datetime import datetime

from .config import Settings
from .openai_client import OpenAIClient
from .search import JST, _extract_after, dedupe_messages, mail_meta, plan_search
from .storage import Storage, StoredMessage

_AGG_INTENT = re.compile(
    r"何件|何通|いくつ|件数|回数|合計|総数|一覧|リスト|全部|全て|すべて|どこ(に|宛|へ)|どちら"
)
# Enumerate ("list each / their contents") — often a follow-up to a count answer.
_ENUM_INTENT = re.compile(r"それぞれ|各(メール|件|通|社)?|一覧|リスト|全部|全て|すべて|中身|内容|明細|\d+\s*件")
_MAIL_WORD = re.compile(r"メール|送っ|送信|受信|届い|やり取り|やりとり|返信|出し")
_ADDR = re.compile(r"[\w.+-]+@[\w.-]+")
_SENT = re.compile(r"送(っ|信|る|り)|出し|出す")
_RECV = re.compile(r"受(信|け)|届い|きた|来た|もらっ|貰っ")
_HEADER_PAIR = re.compile(r'"?([^"<>\n]{1,30})"?\s*[<＜(]\s*([\w.+-]+@[\w.-]+)')


def is_email_aggregate(question: str) -> bool:
    q = question or ""
    return bool((_AGG_INTENT.search(q) or _ENUM_INTENT.search(q)) and _MAIL_WORD.search(q))


def wants_contents(question: str) -> bool:
    """True when the asker wants each email listed (not just a count)."""
    return bool(_ENUM_INTENT.search(question or ""))


def _body_snippet(text: str, limit: int = 160) -> str:
    body = re.sub(r"mailto:[^|>\s]+\|", "", html.unescape(text or ""))
    marker = re.search(r"(本文|内容)\s*[:：]?\s*\*?", body)
    if marker:
        body = body[marker.end():]
    else:  # strip the "…宛に新着メールがありました" wrapper and rule lines
        body = re.sub(r"^.*?新着メールがありました\*?", "", body, flags=re.S)
        body = re.sub(r"-{3,}", " ", body)
    body = body.replace("```", " ")
    body = " ".join(body.split())
    return body[:limit] + ("…" if len(body) > limit else "")


def _direction(question: str) -> str | None:
    q = question or ""
    if _SENT.search(q):
        return "送信"
    if _RECV.search(q):
        return "受信"
    return None  # either direction


def resolve_addresses_for_person(person: str, storage: Storage) -> set[str]:
    """Find the mar5aux email address(es) that belong to `person`.

    `person` may already be an address/handle ("j-naito", "j-naito.p@...") or a
    display name ("内藤淳次郎"). We match it against the display-name/address pairs
    that appear in forwarded-mail headers, so no maintained roster is required.
    """
    person = (person or "").strip()
    if not person:
        return set()
    direct = set(_ADDR.findall(person))
    if direct:
        return direct
    tokens = {t for t in re.split(r"[\s　]+", person) if len(t) >= 2}
    for tok in list(tokens):  # add the surname so "内藤淳次郎" also matches display "内藤"
        if not tok.isascii() and len(tok) >= 2:
            tokens.add(tok[:2])
    if not tokens:
        return set()
    hits: Counter[str] = Counter()
    with storage.connect() as conn:
        rows = conn.execute(
            "SELECT text FROM messages WHERE is_deleted=FALSE AND source_type='bot_message'"
        ).fetchall()
    for row in rows:
        text = html.unescape(dict(row)["text"] or "")
        text = re.sub(r"mailto:[^|>\s]+\|", "", text)  # <mailto:a@b|a@b> -> <a@b>
        for match in _ADDR.finditer(text):
            addr = match.group(0)
            if not addr.lower().endswith("mar5aux.co.jp"):
                continue  # an internal person's own address
            window = text[max(0, match.start() - 40):match.start()]
            low = addr.lower()
            if any((tok in window) or (tok.lower() in low) for tok in tokens):
                hits[addr] += 1
    if not hits:
        return set()
    top = hits.most_common(1)[0][1]
    return {addr for addr, n in hits.items() if n >= max(3, top * 0.3)}


def _fetch_emails(storage: Storage, oldest_ts: str | None, latest_ts: str | None):
    clauses = ["is_deleted=FALSE", "source_type='bot_message'"]
    params: list = []
    if oldest_ts:
        clauses.append("ts::double precision >= %s")
        params.append(float(oldest_ts))
    if latest_ts:
        clauses.append("ts::double precision < %s")
        params.append(float(latest_ts))
    sql = "SELECT ts, channel_name, text, permalink FROM messages WHERE " + " AND ".join(clauses)
    with storage.connect() as conn:
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


class _Row:
    def __init__(self, d):
        self.ts = d["ts"]
        self.channel_name = d["channel_name"]
        self.text = d["text"]
        self.permalink = d.get("permalink")
        self.source_type = "bot_message"


def _date_to_ts(date_str: str | None) -> float | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=JST).timestamp()
    except ValueError:
        return None


def query_emails(
    storage: Storage,
    *,
    person: str = "",
    direction: str = "any",
    counterpart: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    mode: str = "count",
    asker_name: str | None = None,
) -> str:
    """Deterministic count/list of forwarded emails over the whole (deduped) DB.

    A tool the agent calls with explicit parameters — the agent decides person /
    direction / counterpart / period / mode, this just executes and reports.
    `person` is an internal person (name/handle/address); `counterpart` filters the
    other party (name/domain/address substring). direction: sent|received|any.
    mode: count (totals + by-counterpart) | list (each mail with a body snippet).
    """
    person = (person or "").strip()
    if person.lower() in ("me", "私", "自分", "僕", "俺", "わたし") and asker_name:
        person = asker_name
    addresses = resolve_addresses_for_person(person, storage) if person else set()
    addr_low = {a.lower() for a in addresses}
    if person and not addresses:
        return (f"『{person}』の社用メールアドレスをデータから特定できませんでした。"
                f"相手先の会社/ドメインが分かる場合は counterpart で指定してください。")

    oldest = _date_to_ts(start_date)
    latest = _date_to_ts(end_date)
    rows = _fetch_emails(storage, str(oldest) if oldest else None, str(latest) if latest else None)
    deduped = dedupe_messages([_Row(d) for d in rows])

    dir_map = {"sent": "送信", "received": "受信"}
    want_dir = dir_map.get(direction)
    cp_low = (counterpart or "").lower().strip()

    matched = []
    for m in deduped:
        meta = mail_meta(m)
        if not meta:
            continue
        d = meta.get("direction")
        if want_dir and d != want_dir:
            continue
        if d not in ("送信", "受信"):
            continue
        sender = (meta.get("sender") or "").lower()
        recipient = (meta.get("recipient") or "").lower()
        if addr_low:
            if d == "送信" and not any(a in sender for a in addr_low):
                continue
            if d == "受信" and not any(a in recipient for a in addr_low):
                continue
        uptext = html.unescape(m.text or "")
        # counterpart's raw header line (display name + address), so a filter like
        # "tanakaseni" or "田中" matches even though mail_meta only keeps the address.
        raw_cp = (_extract_after(uptext, ("【宛先】", "宛先")) if d == "送信"
                  else _extract_after(uptext, ("【送信元】", "送信元")))
        raw_cp = raw_cp or (meta.get("recipient") if d == "送信" else meta.get("sender")) or ""
        if cp_low and cp_low not in raw_cp.lower() and cp_low not in uptext.lower():
            continue  # match the header line, or fall back to the body (e.g. a signature)
        addr = _ADDR.findall(raw_cp)
        matched.append({
            "ts": m.ts,
            "when": datetime.fromtimestamp(float(m.ts), JST).strftime("%m/%d %H:%M"),
            "counterpart": addr[0] if addr else raw_cp.strip('" <>（）')[:40],
            "subject": " ".join((_extract_after(uptext, ("【件名】", "件名")) or "").split())[:80],
            "snippet": _body_snippet(m.text or ""),
            "permalink": getattr(m, "permalink", None),
        })

    total = len(matched)
    scope = []
    if person:
        scope.append(f"{person}（{'/'.join(sorted(addresses))}）" if addresses else person)
    scope.append({"sent": "送信", "received": "受信"}.get(direction, "送受信"))
    if counterpart:
        scope.append(f"相手={counterpart}")
    if start_date or end_date:
        scope.append(f"期間 {start_date or '〜'}〜{end_date or ''}（終端含まず）")
    header = " / ".join(scope)

    by_cp = Counter(it["counterpart"] for it in matched)
    label = "差出人" if direction == "received" else "宛先"

    if mode == "list":
        lines = [f"[{header}] 該当 {total}件（重複除去後）",
                 f"{label}別: " + "、".join(f"{cp} {n}件" for cp, n in by_cp.most_common()), ""]
        for i, it in enumerate(sorted(matched, key=lambda x: float(x["ts"])), 1):
            link = f"  <{it['permalink']}|開く>" if it.get("permalink") else ""
            lines.append(f"{i}. {it['when']} → {it['counterpart']} ｜ 件名: {it['subject'] or '(なし)'}{link}")
            if it["snippet"]:
                lines.append(f"    {it['snippet']}")
        return "\n".join(lines)

    lines = [f"[{header}] 合計 {total}件（重複除去後）", f"{label}別:"]
    for cp, n in by_cp.most_common():
        lines.append(f"  {cp}: {n}件")
    # Reference links: one representative message per counterpart (up to 6).
    refs = []
    seen_cp: set[str] = set()
    for it in sorted(matched, key=lambda x: float(x["ts"])):
        if it["counterpart"] in seen_cp or not it.get("permalink"):
            continue
        seen_cp.add(it["counterpart"])
        refs.append(f"<{it['permalink']}|{it['when']} {it['counterpart']}>")
        if len(refs) >= 6:
            break
    if refs:
        lines.append("\n参照リンク:")
        lines.extend(f"{i}. {r}" for i, r in enumerate(refs, 1))
    return "\n".join(lines)


def run_email_aggregate(
    question: str,
    storage: Storage,
    openai_client: OpenAIClient,
    asker_name: str | None = None,
) -> str | None:
    """Return a deterministic count/enumeration answer, or None if not applicable."""
    if not is_email_aggregate(question):
        return None

    plan = plan_search(question, openai_client)
    # who: explicit person from the planner, or first-person -> the asker
    person = ""
    if plan.person_names:
        person = plan.person_names[0]
    elif re.search(r"私|自分|僕|俺|わたし|my|me", question, re.IGNORECASE) and asker_name:
        person = asker_name
    else:
        found = _ADDR.findall(question) or re.findall(r"[a-z]-[a-z]+", question)
        if found:
            person = found[0]
    if not person:
        return None

    addresses = resolve_addresses_for_person(person, storage)
    if not addresses:
        return None
    addr_low = {a.lower() for a in addresses}

    direction = _direction(question)
    rows = _fetch_emails(storage, plan.oldest_ts if plan.date_intent else None,
                         plan.latest_ts if plan.date_intent else None)
    deduped = dedupe_messages([_Row(d) for d in rows])

    matched = []
    for m in deduped:
        meta = mail_meta(m)
        if not meta:
            continue
        d = meta.get("direction")
        if direction and d != direction:
            continue
        sender = (meta.get("sender") or "").lower()
        recipient = (meta.get("recipient") or "").lower()
        if d == "送信" and any(a in sender for a in addr_low):
            counterpart = meta.get("recipient") or "(不明)"
        elif d == "受信" and any(a in recipient for a in addr_low):
            counterpart = meta.get("sender") or "(不明)"
        else:
            continue
        cp = _ADDR.findall(counterpart)
        subject = " ".join((_extract_after(html.unescape(m.text or ""), ("【件名】", "件名")) or "").split())[:80]
        matched.append({
            "ts": m.ts,
            "when": datetime.fromtimestamp(float(m.ts), JST).strftime("%m/%d %H:%M"),
            "counterpart": cp[0] if cp else counterpart.strip('" <>（）'),
            "subject": subject,
            "snippet": _body_snippet(m.text or ""),
        })

    total = len(matched)
    period = ""
    if plan.date_intent and plan.start_date and plan.end_date:
        period = f"{plan.start_date}〜{plan.end_date}（終端は含まず）の"
    dir_word = {"送信": "送信した", "受信": "受信した"}.get(direction, "やり取りした")
    who = person if not addresses else f"{person}（{'/'.join(sorted(addresses))}）"

    if wants_contents(question):
        lines = [f"{period}{who} が{dir_word}メール（重複除去後 *{total}件*）の内容は以下です。"]
        for i, item in enumerate(sorted(matched, key=lambda x: float(x["ts"])), 1):
            head = f"\n*{i}. {item['when']} → {item['counterpart']}*"
            if item["subject"]:
                head += f"\n件名: {item['subject']}"
            if item["snippet"]:
                head += f"\n{item['snippet']}"
            lines.append(head)
        lines.append("\n（ミラー／連投の重複は除いています。本文は各メールの冒頭抜粋です）")
        return "\n".join(lines)

    by_cp = Counter(item["counterpart"] for item in matched)
    lines = [f"{period}{who} が{dir_word}メールは、重複除去後 *{total}件* です。"]
    if by_cp:
        label = "宛先" if direction != "受信" else "差出人"
        lines.append(f"\n*{label}別*")
        for cp, n in by_cp.most_common():
            lines.append(f"• {cp}: {n}件")
    lines.append("\n（同じメールが複数メールボックスにミラー／連投された重複は除いて集計しています）")
    return "\n".join(lines)
