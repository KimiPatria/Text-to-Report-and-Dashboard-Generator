"""Pipeline error-handling utilities for the EPMS AI."""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DOMAIN_KEYWORDS = frozenset({
    "harvest", "block", "division", "estate", "employee", "attendance",
    "bunches", "ffb", "palm", "yield", "tonnage", "worker", "mandor",
    "workplan", "overtime",
})

# High-traffic daily transaction tables — descriptions drive the Case B suggestion list.
_SUGGESTION_TABLES = [
    "t_oph",
    "t_attendance",
    "t_work_assignment",
    "t_harvesting_plan",
    "t_workdone",
]


def _build_suggestions() -> str:
    """Pull the first descriptive clause from each suggestion table's metadata."""
    path = Path(__file__).parent / "schema_metadata.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("tables", {})
        phrases: list[str] = []
        for name in _SUGGESTION_TABLES:
            entry = meta.get(name)
            if not entry:
                continue
            desc = entry.get("description", "")
            # Take the first clause before the first period or comma, lowercased
            first = desc.split(".")[0].split(",")[0].strip().lower()
            if first:
                phrases.append(first)
        if phrases:
            return ", ".join(phrases)
    except Exception as exc:
        log.warning("Cannot load suggestions from schema_metadata.json: %s", exc)
    return "harvest bunches, employee attendance, work assignments, or harvesting plans"


CASE_B_SUGGESTIONS: str = _build_suggestions()

_MSG_OFF_DOMAIN = (
    "Sorry, I cannot answer this because I cannot find relevant data. "
    "Please try a different query related to plantation operations."
)

_MSG_NO_DATA = (
    "Sorry, I cannot generate this report because I cannot find the required data. "
    f"Please try rephrasing your query, or try asking about: {CASE_B_SUGGESTIONS}."
)


def error_response(kind: str, used_llm_fallback: bool = False) -> dict:
    """Build the standard pipeline error dict (returned as HTTP 200 JSON)."""
    msg = _MSG_OFF_DOMAIN if kind == "off_domain" else _MSG_NO_DATA
    return {
        "answer": msg,
        "sql": None,
        "rows": [],
        "columns": [],
        "error": kind,
        "used_llm_fallback": used_llm_fallback,
    }


def is_off_domain(query: str) -> bool:
    """Return True only when the query has no connection to plantation operations.

    Fast path: any domain keyword match → return False immediately (no LLM call).
    Slow path: routing-model YES/NO classification for edge cases.
    """
    tokens = set(query.lower().split())
    if tokens & _DOMAIN_KEYWORDS:
        return False  # keyword hit — definitely on-domain

    # No keyword match — ask the routing model
    try:
        import llm as _llm
        from config import ROUTING_MODEL

        client = _llm._groq_client
        if not client:
            log.warning("Routing LLM unavailable for domain check — assuming on-domain")
            return False

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a domain classifier for a palm oil plantation management system. "
                    'Respond ONLY with JSON: {"related": true} or {"related": false}. '
                    "Answer true if the query is about palm oil plantation, agriculture, "
                    "estate management, HR, harvesting, or FFB. "
                    "Answer false only if completely unrelated to these topics."
                ),
            },
            {"role": "user", "content": query},
        ]
        completion = client.chat.completions.create(
            model=ROUTING_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
        related = bool(data.get("related", True))
        log.info("[domain_check] query=%r related=%s", query[:80], related)
        return not related
    except Exception as exc:
        log.warning("Domain check LLM failed: %s — assuming on-domain", exc)
        return False  # safe fallback: never incorrectly reject a plantation query
