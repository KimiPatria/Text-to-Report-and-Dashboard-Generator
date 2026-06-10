"""
EPMS Data Chatbot — FastAPI router
Mounted at /chat on dashboard_server.py (port 8001).

POST /chat/message — conversational Q&A backed by the EPMS database.
Pipeline: retrieve tables → generate SQL → execute → reason → respond.
"""

import decimal
import logging
import time
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import engine
from error_handler import is_off_domain
from llm import call_llm, get_provider, PROVIDER_LABELS
from prompt_builder import build_sql_messages, extract_sql
from query_history import record_query
from retrieval import retrieve_tables, retrieve_examples
from sql_validator import validate, ensure_limit

log = logging.getLogger("epms-chat")

import json

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
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
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


# ── endpoint ───────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatResponse)
def chat_message(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "message must not be empty"})

    t0 = time.monotonic()
    log.info('[chat] message="%s"', message[:120])

    # ── 1. Retrieve tables ────────────────────────────────────────────────
    table_cards, used_llm_fallback = retrieve_tables(message, k=5)
    table_names = [c.name for c in table_cards]
    log.info("[chat] tables: %s (llm_fallback=%s)", table_names, used_llm_fallback)

    if not table_cards:
        kind = "off_domain" if is_off_domain(message) else "no_data"
        log.info("[chat] no tables — %s", kind)
        answer = _OFF_DOMAIN_ANSWER if kind == "off_domain" else _NO_DATA_ANSWER
        return ChatResponse(
            answer=answer,
            used_llm_fallback=used_llm_fallback,
            error_type=kind,
        )

    # ── 2. Few-shot examples ──────────────────────────────────────────────
    try:
        examples = retrieve_examples(message, k=2)
    except Exception:
        examples = []

    # ── 3. Generate SQL ───────────────────────────────────────────────────
    sql_messages = build_sql_messages(message, table_cards, examples)
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
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        )

    # ── 4. Validate SQL ───────────────────────────────────────────────────
    ok, reason = validate(sql)
    if not ok:
        log.info("[chat] SQL validation failed: %s", reason)
        return ChatResponse(
            answer=_SQL_FAIL_ANSWER,
            sql=sql,
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        )

    # ── 5. Execute (with one retry on SQL error) ──────────────────────────
    columns, rows, err = _execute_safe(sql)
    if err:
        log.info("[chat] SQL exec error — retrying once: %s", err[:120])
        retry_messages = build_sql_messages(
            message, table_cards, examples,
            prior_error=err, prior_sql=sql,
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
            sql=sql,
            tables_used=table_names,
            provider=provider_used,
            provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
            used_llm_fallback=used_llm_fallback,
            error_type="no_data",
        )

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
        sql=sql,
        columns=columns,
        rows=rows[:100],
        row_count=len(rows),
        tables_used=table_names,
        provider=provider_used,
        provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
        used_llm_fallback=used_llm_fallback,
    )
