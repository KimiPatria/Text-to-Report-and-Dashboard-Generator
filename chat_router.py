"""
EPMS Data Chatbot — FastAPI router
Mounted at /chat on dashboard_server.py (port 8001).

POST /chat/message — conversational Q&A backed by the EPMS database.
Session-scoped context: a lightweight router decides per message whether to
answer from conversation history (no SQL) or run the full pipeline:
retrieve tables → generate SQL → execute → reason → respond.
"""

import decimal
import json
import logging
import time
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from chat_history import (
    add_turn,
    delete_session,
    ensure_session,
    get_turns,
    list_sessions,
)
from config import (
    engine,
    CHAT_HISTORY_DATA_TURNS,
    CHAT_HISTORY_MAX_ROWS,
    CHAT_HISTORY_TEXT_TURNS,
    ROUTING_MODEL,
)
from error_handler import is_off_domain
from llm import call_llm, get_provider, PROVIDER_LABELS
from prompt_builder import build_sql_messages, extract_sql
from query_history import record_query
from retrieval import retrieve_tables, retrieve_examples
from sql_validator import validate, ensure_limit

log = logging.getLogger("epms-chat")

_CHAT_ANSWER_SYSTEM_PROMPT = (
    "You are a knowledgeable data analyst for an EPMS palm-oil plantation. "
    "The user asked a question, SQL was run against the database, and the results are provided. "
    "Respond conversationally in plain English — let the data shape your answer naturally. "
    "Do not follow a fixed structure. Adapt to what the data actually shows:\n"
    "- A simple total or count: state it directly, add a brief sentence of context.\n"
    "- A comparison or ranking: highlight what stands out and why it matters.\n"
    "- A trend over time: describe the direction and any notable points.\n"
    "- Complex or multi-column results: walk through the most important findings naturally.\n"
    "Always cite specific numbers with units. Be concise — no padding, no preamble, "
    "no closing remarks like 'I hope this helps'. If the data is empty, say so plainly."
)

_CONTEXT_ANSWER_SYSTEM_PROMPT = (
    "You are a knowledgeable data analyst for an EPMS palm-oil plantation. "
    "Answer the user's follow-up question using ONLY the conversation history and the "
    "previously fetched data provided below — no new database query will be run. "
    "Cite specific numbers with units. Be concise — no padding, no preamble. "
    "If the provided data cannot fully answer the question, say what is missing and "
    "suggest asking it as a new question."
)

_ROUTER_SYSTEM_PROMPT = (
    "You route questions for a plantation-data chatbot. Given the recent conversation "
    "and a new question, decide whether it can be answered using only that conversation "
    "(including any data rows already fetched), or whether it needs a fresh database query. "
    'Reply ONLY with JSON: {"route": "context"} or {"route": "sql"}.\n'
    'Pick "context" when the question refers to, clarifies, reformats, or asks for '
    "calculations or explanations about information already shown.\n"
    'Pick "sql" when it needs new or different data: another metric, entity, date range, '
    "filter, or more rows than were fetched."
)


def _build_answer_messages(
    question: str,
    sql: str,
    rows: list[dict],
    columns: list[str],
    stats: dict,
) -> list[dict]:
    preview = rows[:50]
    preview_json = json.dumps(preview, default=str, ensure_ascii=False)
    stats_json   = json.dumps(stats,   default=str, ensure_ascii=False)
    return [
        {"role": "system", "content": _CHAT_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"SQL executed:\n{sql}\n\n"
                f"Columns: {columns}\n"
                f"Total rows returned: {len(rows)}\n"
                f"Summary stats: {stats_json}\n\n"
                f"Rows (first 50): {preview_json}"
            ),
        },
    ]


router = APIRouter(prefix="/chat", tags=["chat"])


# ── request / response models ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: list[ChatMessage] = []  # legacy field — server-side history is authoritative


class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None
    route: str = "sql"          # "sql" | "context"
    sql: Optional[str] = None
    columns: list[str] = []
    rows: list[dict] = []
    row_count: int = 0
    tables_used: list[str] = []
    provider: Optional[str] = None
    provider_label: Optional[str] = None
    used_llm_fallback: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None


# ── helpers ────────────────────────────────────────────────────────────────

def _coerce(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return str(v)
    return v


def _execute_safe(
    sql: str, max_rows: int = 200
) -> tuple[list[str], list[dict], Optional[str]]:
    try:
        limited = ensure_limit(sql, max_rows)
        with engine.connect() as conn:
            result = conn.execute(text(limited))
            columns = list(result.keys())
            rows = [
                {k: _coerce(v) for k, v in dict(r._mapping).items()}
                for r in result.fetchall()
            ]
        return columns, rows, None
    except SQLAlchemyError as exc:
        err = str(exc.orig) if hasattr(exc, "orig") and exc.orig else str(exc)
        log.warning("[chat] SQL error: %s", err[:300])
        return [], [], err
    except Exception as exc:
        log.warning("[chat] unexpected error: %s", exc)
        return [], [], str(exc)


def _compute_stats(columns: list[str], rows: list[dict]) -> dict:
    stats: dict = {}
    for col in columns:
        vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
        if vals:
            stats[col] = {
                "count": len(vals),
                "sum": round(sum(vals), 4),
                "min": min(vals),
                "max": max(vals),
                "avg": round(sum(vals) / len(vals), 4),
            }
    return stats


_OFF_DOMAIN_ANSWER = (
    "I can only answer questions about your EPMS plantation data — "
    "harvest records, employee information, blocks, estates, and related topics. "
    "Please try asking something related to your plantation operations."
)

_NO_DATA_ANSWER = (
    "I wasn't able to find relevant data for that question. "
    "Try rephrasing or asking about a specific estate, block, employee, or harvest period."
)

_SQL_FAIL_ANSWER = (
    "I found relevant data tables but couldn't produce a working query for that question. "
    "Try rephrasing with more specific terms, such as an estate code, date range, or metric name."
)


# ── conversational context ─────────────────────────────────────────────────

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _route_followup(message: str, turns: list[dict]) -> str:
    """Decide 'context' vs 'sql' with the cheap routing model.

    Sees text Q&A only (answers truncated) plus a one-line note about which
    turns still have data rows available. Falls back to 'sql' on any failure.
    """
    lines: list[str] = []
    for t in turns[-CHAT_HISTORY_TEXT_TURNS:]:
        lines.append(f"Q: {_truncate(t['question'], 200)}")
        lines.append(f"A: {_truncate(t['answer'], 240)}")
        if t["rows"]:
            lines.append(
                f"[data fetched for this turn: columns {t['columns']}, "
                f"{len(t['rows'])} of {t['row_count']} rows available]"
            )
    user_content = "\n".join(lines) + f"\n\nNew question: {message}"

    try:
        import llm as _llm
        client = _llm._groq_client
        if not client:
            return "sql"
        completion = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        u = completion.usage
        log.info(
            "token_usage phase=chat_route model=%s input=%d output=%d",
            ROUTING_MODEL,
            u.prompt_tokens if u else 0,
            u.completion_tokens if u else 0,
        )
        route = json.loads(raw).get("route", "sql")
        return route if route in ("context", "sql") else "sql"
    except Exception as exc:
        log.warning("[chat] follow-up router failed: %s — defaulting to sql", exc)
        return "sql"


def _build_context_messages(message: str, turns: list[dict]) -> list[dict]:
    """Prompt for answering from history. Tiered payload: text Q&A for recent
    turns, full rows only for the last CHAT_HISTORY_DATA_TURNS data-bearing
    turns, capped at CHAT_HISTORY_MAX_ROWS rows each."""
    text_turns = turns[-CHAT_HISTORY_TEXT_TURNS:]
    data_turns = [t for t in turns if t["rows"]][-CHAT_HISTORY_DATA_TURNS:]

    parts: list[str] = ["Conversation so far:"]
    for t in text_turns:
        parts.append(f"Q: {_truncate(t['question'], 300)}")
        # Rows for data-bearing turns are appended below (with their SQL);
        # for row-less turns the SQL alone still grounds meta-questions.
        if t["sql"] and t not in data_turns:
            parts.append(f"SQL used (returned {t['row_count']} rows): {t['sql']}")
        parts.append(f"A: {_truncate(t['answer'], 500)}")

    for t in data_turns:
        rows = t["rows"][:CHAT_HISTORY_MAX_ROWS]
        rows_json  = json.dumps(rows, default=str, ensure_ascii=False)
        stats_json = json.dumps(t["stats"], default=str, ensure_ascii=False)
        parts.append(
            f"\nData fetched earlier for: {_truncate(t['question'], 200)}\n"
            f"SQL: {t['sql']}\n"
            f"Columns: {t['columns']}\n"
            f"Summary stats (over all {t['row_count']} rows): {stats_json}\n"
            f"Rows (first {len(rows)} of {t['row_count']}): {rows_json}"
        )

    parts.append(f"\nFollow-up question: {message}")
    return [
        {"role": "system", "content": _CONTEXT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _sql_history_block(turns: list[dict]) -> str:
    """Text-only history for the SQL writer — last 2 turns, question + SQL."""
    lines: list[str] = []
    for t in turns[-2:]:
        lines.append(f"Q: {_truncate(t['question'], 200)}")
        if t["sql"]:
            lines.append(f"SQL: {t['sql']}")
        else:
            lines.append(f"A: {_truncate(t['answer'], 160)}")
    return "\n".join(lines)


def _answer_from_context(
    sid: str, message: str, turns: list[dict], t0: float
) -> ChatResponse:
    messages = _build_context_messages(message, turns)
    answer_raw, provider_used, usage = call_llm(messages, max_tokens=600, temperature=0)
    log.info(
        "[chat] context-answer provider=%s input=%d output=%d",
        provider_used, usage["input_tokens"], usage["output_tokens"],
    )
    answer = answer_raw.strip() or _NO_DATA_ANSWER
    log.info("[chat] done %.2fs (from context)", time.monotonic() - t0)
    return ChatResponse(
        answer=answer,
        session_id=sid,
        route="context",
        provider=provider_used,
        provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
    )


# ── endpoint ───────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatResponse)
def chat_message(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "message must not be empty"})

    t0 = time.monotonic()
    sid = ensure_session(req.session_id, message)
    turns = get_turns(sid, limit=CHAT_HISTORY_TEXT_TURNS + CHAT_HISTORY_DATA_TURNS)
    log.info('[chat] message="%s" session=%s turns=%d', message[:120], sid[:8], len(turns))

    # ── 0. Route: answer from context or hit the database? ───────────────
    if turns and _route_followup(message, turns) == "context":
        resp = _answer_from_context(sid, message, turns, t0)
        add_turn(sid, message, resp.answer, route="context")
        return resp

    resp, stats = _run_sql_pipeline(sid, message, turns, t0)
    add_turn(
        sid, message, resp.answer,
        sql=resp.sql if not resp.error_type else None,
        columns=resp.columns, rows=resp.rows,
        row_count=resp.row_count, stats=stats,
        route="sql",
    )
    return resp


def _run_sql_pipeline(
    sid: str, message: str, turns: list[dict], t0: float
) -> tuple[ChatResponse, Optional[dict]]:
    # ── 1. Retrieve tables (augmented with the prior question on follow-ups) ─
    retrieval_query = message
    if turns:
        retrieval_query = f"{turns[-1]['question']}\n{message}"
    table_cards, used_llm_fallback = retrieve_tables(retrieval_query, k=5)
    table_names = [c.name for c in table_cards]
    log.info("[chat] tables: %s (llm_fallback=%s)", table_names, used_llm_fallback)

    if not table_cards:
        kind = "off_domain" if is_off_domain(message) else "no_data"
        log.info("[chat] no tables — %s", kind)
        answer = _OFF_DOMAIN_ANSWER if kind == "off_domain" else _NO_DATA_ANSWER
        return ChatResponse(
            answer=answer,
            session_id=sid,
            used_llm_fallback=used_llm_fallback,
            error_type=kind,
        ), None

    # ── 2. Few-shot examples ──────────────────────────────────────────────
    try:
        examples = retrieve_examples(message, k=2)
    except Exception:
        examples = []

    # ── 3. Generate SQL (text-only history so follow-ups resolve) ────────
    history_block = _sql_history_block(turns) if turns else None
    sql_messages = build_sql_messages(
        message, table_cards, examples, history_block=history_block
    )
    raw_sql, provider_used, usage = call_llm(sql_messages, max_tokens=800, temperature=0)
    log.info(
        "[chat] SQL gen provider=%s input=%d output=%d",
        provider_used, usage["input_tokens"], usage["output_tokens"],
    )

    sql = extract_sql(raw_sql)
    if not sql:
        log.info("[chat] no SQL extracted")
        return ChatResponse(
            answer=_NO_DATA_ANSWER,
            session_id=sid,
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        ), None

    # ── 4. Validate SQL ───────────────────────────────────────────────────
    ok, reason = validate(sql)
    if not ok:
        log.info("[chat] SQL validation failed: %s", reason)
        return ChatResponse(
            answer=_SQL_FAIL_ANSWER,
            session_id=sid,
            sql=sql,
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        ), None

    # ── 5. Execute (with one retry on SQL error) ──────────────────────────
    columns, rows, err = _execute_safe(sql)
    if err:
        log.info("[chat] SQL exec error — retrying once: %s", err[:120])
        retry_messages = build_sql_messages(
            message, table_cards, examples,
            prior_error=err, prior_sql=sql,
            history_block=history_block,
        )
        raw_retry, provider_used, usage = call_llm(retry_messages, max_tokens=800, temperature=0)
        sql_retry = extract_sql(raw_retry)
        if sql_retry:
            ok2, _ = validate(sql_retry)
            if ok2:
                columns, rows, err = _execute_safe(sql_retry)
                if not err:
                    sql = sql_retry

    if err:
        log.info("[chat] SQL still failing after retry")
        record_query(message, sql, provider_used, table_names, success=False, error=err)
        return ChatResponse(
            answer=_SQL_FAIL_ANSWER,
            session_id=sid,
            sql=sql,
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        ), None

    # ── 6. Compute stats & build answer ──────────────────────────────────
    stats = _compute_stats(columns, rows)
    answer_messages = _build_answer_messages(message, sql, rows, columns, stats)
    answer_raw, provider_used, usage2 = call_llm(
        answer_messages, max_tokens=600, temperature=0
    )
    log.info(
        "[chat] reasoning provider=%s input=%d output=%d",
        provider_used, usage2["input_tokens"], usage2["output_tokens"],
    )

    answer = answer_raw.strip() or _NO_DATA_ANSWER

    # ── 7. Record to history & refresh index ─────────────────────────────
    record_query(message, sql, provider_used, table_names, success=True)
    try:
        from retrieval import refresh_query_history_index
        refresh_query_history_index()
    except Exception:
        pass

    elapsed = time.monotonic() - t0
    log.info("[chat] done %.2fs rows=%d", elapsed, len(rows))

    return ChatResponse(
        answer=answer,
        session_id=sid,
        sql=sql,
        columns=columns,
        rows=rows[:100],
        row_count=len(rows),
        tables_used=table_names,
        provider=provider_used,
        provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
        used_llm_fallback=used_llm_fallback,
    ), stats


# ── session endpoints ──────────────────────────────────────────────────────

@router.get("/sessions")
def sessions_index():
    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
def session_detail(session_id: str):
    return {"id": session_id, "turns": get_turns(session_id, limit=100)}


@router.delete("/sessions/{session_id}")
def session_remove(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "session not found"})
    return {"ok": True}
