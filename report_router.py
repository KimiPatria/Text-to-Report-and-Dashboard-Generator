"""
EPMS Report Generator — FastAPI router
Mounted at /report on dashboard_server.py (port 8001).

Two endpoints:
  POST /report/preset   — fixed SQL, no LLM, zero token cost
  POST /report/generate — full RAG → SQL → execute pipeline
"""

import logging
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import engine
from llm import call_llm, call_normalization_llm
from query_normalizer import normalize_saved_query
from metadata_loader import compact_ddl_for
from prompt_builder import build_sql_messages, extract_sql
from retrieval import retrieve_tables
from saved_queries import (
    delete_template,
    get_all_templates,
    get_template,
    init_db as _init_saved_db,
    rename_template,
    save_template,
    update_template_normalization,
)
from sql_validator import ensure_limit, validate

_init_saved_db()

log = logging.getLogger("epms-report")

router = APIRouter(prefix="/report", tags=["report"])


# ── Pydantic models ────────────────────────────────────────────────────────

class PresetRequest(BaseModel):
    preset: Literal["general", "employee", "harvest"]
    days: int = 30


class GenerateReportRequest(BaseModel):
    query: str


class RunSavedRequest(BaseModel):
    sql: str
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ReportSection(BaseModel):
    title: str
    columns: list[str] = []
    rows: list[dict] = []
    sql: Optional[str] = None
    error: Optional[str] = None


class ReportResponse(BaseModel):
    title: str
    preset: Optional[str] = None
    sections: list[ReportSection]
    generated_sql: Optional[str] = None
    provider: Optional[str] = None
    used_llm_fallback: Optional[bool] = None


class NormalizeTemplateRequest(BaseModel):
    sql: str
    query: str = ""
    name: str


class RunTemplateRequest(BaseModel):
    saved_query_id: str
    date_from: str
    date_to: str
    granularity: Optional[Literal["day", "week", "month"]] = None


class MigrateTemplatesRequest(BaseModel):
    templates: list[dict]


class RenameTemplateRequest(BaseModel):
    name: str


# ── SQL table-name extractor (for DDL context) ────────────────────────────

_TABLE_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?',
    re.IGNORECASE,
)


def _extract_table_names(sql: str) -> list[str]:
    return list({m.group(1) for m in _TABLE_RE.finditer(sql)})


# ── DB helper ──────────────────────────────────────────────────────────────

def _coerce(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return str(v)
    return v


def _execute(
    sql: str,
    params: dict | None = None,
    max_rows: int = 1000,
) -> tuple[list[str], list[dict], Optional[str]]:
    try:
        limited = ensure_limit(sql, max_rows)
        with engine.connect() as conn:
            result = conn.execute(text(limited), params or {})
            columns = list(result.keys())
            rows = [
                {k: _coerce(v) for k, v in dict(r._mapping).items()}
                for r in result.fetchall()
            ]
        return columns, rows, None
    except SQLAlchemyError as exc:
        err = str(exc.orig) if hasattr(exc, "orig") and exc.orig else str(exc)
        log.warning("[report] SQL error: %s", err[:300])
        return [], [], err
    except Exception as exc:
        log.warning("[report] unexpected error: %s", exc)
        return [], [], str(exc)


# ── Preset SQL definitions ─────────────────────────────────────────────────
# Each entry: (section_title, sql_string)
# Date params :start_date and :end_date are injected for sections that need them.

_PRESETS: dict[str, list[tuple[str, str]]] = {

    "general": [
        (
            "Workforce Summary by Estate",
            """
SELECT
    e.employee_estate_code                                              AS "Estate",
    COUNT(*)                                                            AS "Total Employees",
    COUNT(*) FILTER (WHERE e.employee_status = 'A')                    AS "Active",
    COUNT(*) FILTER (WHERE e.employee_status != 'A')                   AS "Inactive",
    COUNT(*) FILTER (WHERE e.employee_job_type ILIKE '%harvest%'
                        OR e.employee_job_type ILIKE '%panen%')        AS "Harvesters"
FROM m_employee e
GROUP BY e.employee_estate_code
ORDER BY e.employee_estate_code
""",
        ),
        (
            "Block Inventory by Maturity State",
            """
SELECT
    b."ESTNR"                                AS "Estate",
    b."BSTATE"                               AS "Maturity",
    COUNT(*)                                 AS "Blocks",
    ROUND(COALESCE(SUM(b."BHA"), 0)::numeric, 2)  AS "Hectares (Ha)",
    COALESCE(SUM(b."POINT"), 0)              AS "Total Palms"
FROM "ZEPMS_EM_BLOCK_OUT" b
GROUP BY b."ESTNR", b."BSTATE"
ORDER BY b."ESTNR", b."BSTATE"
""",
        ),
        (
            "Harvest Output Summary (Date Range)",
            """
SELECT
    o.oph_estate_code                          AS "Estate",
    o.oph_division_code                        AS "Division",
    COUNT(*)                                   AS "OPH Records",
    SUM(o.bunches_total)                       AS "Total Bunches",
    SUM(o.bunches_ripe)                        AS "Ripe",
    SUM(o.bunches_unripe)                      AS "Unripe",
    SUM(o.bunches_overripe)                    AS "Overripe",
    ROUND(
        CASE
            WHEN SUM(o.bunches_total) > 0
            THEN SUM(o.bunches_ripe) * 100.0 / SUM(o.bunches_total)
            ELSE 0
        END::numeric, 1
    )                                          AS "Ripeness %"
FROM t_oph o
WHERE o.oph_created_date >= :start_date
  AND o.oph_created_date <= :end_date
GROUP BY o.oph_estate_code, o.oph_division_code
ORDER BY o.oph_estate_code, o.oph_division_code
""",
        ),
    ],

    "employee": [
        (
            "Employee Directory",
            """
SELECT
    e.employee_code                AS "Code",
    e.employee_name                AS "Name",
    e.employee_estate_code         AS "Estate",
    e.employee_division_code       AS "Division",
    e.employee_job_code            AS "Job Code",
    e.employee_job_type            AS "Job Type",
    e.employee_status              AS "Status"
FROM m_employee e
ORDER BY e.employee_estate_code, e.employee_division_code, e.employee_name
""",
        ),
        (
            "Attendance Summary by Employee (Date Range)",
            """
SELECT
    e.employee_code                                                        AS "Code",
    e.employee_name                                                        AS "Name",
    e.employee_estate_code                                                 AS "Estate",
    e.employee_division_code                                               AS "Division",
    e.employee_job_type                                                    AS "Job Type",
    COUNT(a.attendance_id)                                                 AS "Days Recorded",
    COUNT(a.attendance_id) FILTER (WHERE a.attendance_is_closed = 1)      AS "Days Closed"
FROM m_employee e
LEFT JOIN t_attendance a
       ON a.attendance_employee_code = e.employee_code
      AND a.attendance_date >= :start_date
      AND a.attendance_date <= :end_date
WHERE e.employee_status = 'A'
GROUP BY
    e.employee_code, e.employee_name,
    e.employee_estate_code, e.employee_division_code, e.employee_job_type
ORDER BY e.employee_estate_code, e.employee_division_code, e.employee_name
""",
        ),
    ],

    "harvest": [
        (
            "Daily Harvest by Block (Date Range)",
            """
SELECT
    o.oph_created_date             AS "Date",
    o.oph_estate_code              AS "Estate",
    o.oph_division_code            AS "Division",
    o.oph_block_code               AS "Block",
    COUNT(*)                       AS "OPH Count",
    SUM(o.bunches_total)           AS "Total Bunches",
    SUM(o.bunches_ripe)            AS "Ripe",
    SUM(o.bunches_unripe)          AS "Unripe",
    SUM(o.bunches_overripe)        AS "Overripe",
    SUM(o.bunches_rotten)          AS "Rotten"
FROM t_oph o
WHERE o.oph_created_date >= :start_date
  AND o.oph_created_date <= :end_date
GROUP BY
    o.oph_created_date, o.oph_estate_code,
    o.oph_division_code, o.oph_block_code
ORDER BY o.oph_estate_code, o.oph_division_code, o.oph_created_date
""",
        ),
        (
            "Average Bunch Weight by Block (Date Range)",
            """
SELECT
    a.abw_estate_code                                    AS "Estate",
    a.abw_block_code                                     AS "Block",
    a.abw_year                                           AS "Year",
    a.abw_month                                          AS "Month",
    ROUND(AVG(a.abw_bunch_weight)::numeric, 2)           AS "Avg Bunch Weight (kg)",
    COUNT(*)                                             AS "Samples"
FROM t_abw a
WHERE a.abw_sample_date >= :start_date
  AND a.abw_sample_date <= :end_date
GROUP BY a.abw_estate_code, a.abw_block_code, a.abw_year, a.abw_month
ORDER BY a.abw_estate_code, a.abw_block_code, a.abw_year, a.abw_month
""",
        ),
    ],
}

_PRESET_TITLES = {
    "general":  "General Overview Report",
    "employee": "Employee Report",
    "harvest":  "Harvest Report",
}


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/preset", response_model=ReportResponse)
def generate_preset(req: PresetRequest) -> ReportResponse:
    days = max(7, min(req.days, 90))
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)
    params   = {"start_date": str(start_dt), "end_date": str(end_dt)}

    title = f"{_PRESET_TITLES[req.preset]} — Last {days} Days ({start_dt} to {end_dt})"

    sections: list[ReportSection] = []
    for section_title, sql in _PRESETS[req.preset]:
        ok, reason = validate(sql)
        if not ok:
            sections.append(ReportSection(title=section_title, error=f"SQL validation: {reason}"))
            continue
        cols, rows, err = _execute(sql, params)
        sections.append(ReportSection(
            title=section_title,
            columns=cols,
            rows=rows,
            sql=sql.strip(),
            error=err,
        ))
        log.info("[report/preset] %s → %d rows", section_title, len(rows))

    return ReportResponse(title=title, preset=req.preset, sections=sections)


@router.post("/generate", response_model=ReportResponse)
def generate_text_report(req: GenerateReportRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    log.info('[report/generate] query="%s"', query)

    from error_handler import error_response as _err_resp, is_off_domain

    # 1. Retrieve relevant tables via hybrid retrieval
    table_cards, used_llm_fallback = retrieve_tables(query, k=5)
    table_names = [c.name for c in table_cards]
    log.info("[report/generate] tables: %s (llm_fallback=%s)", table_names, used_llm_fallback)

    if not table_cards:
        kind = "off_domain" if is_off_domain(query) else "no_data"
        log.info("[report/generate] no tables retrieved — returning %s", kind)
        return JSONResponse(content=_err_resp(kind, used_llm_fallback))

    # 2. Few-shot examples from history
    try:
        from retrieval import retrieve_examples
        examples = retrieve_examples(query, k=2)
    except Exception:
        examples = []

    # 3. Build prompt and call LLM
    messages = build_sql_messages(query, table_cards, examples)
    raw_text, provider_used, usage = call_llm(messages, max_tokens=600, temperature=0)
    log.info(
        "[report/generate] provider=%s input=%d output=%d",
        provider_used, usage["input_tokens"], usage["output_tokens"],
    )

    # 4. Extract SQL
    sql = extract_sql(raw_text)
    if not sql:
        log.info("[report/generate] no SQL extracted")
        return JSONResponse(content=_err_resp("no_data", used_llm_fallback))

    # 5. Validate SQL
    ok, reason = validate(sql)
    if not ok:
        log.info("[report/generate] SQL validation failed: %s", reason)
        resp = _err_resp("no_data", used_llm_fallback)
        resp["generated_sql"] = sql
        return JSONResponse(content=resp)

    # 6. Execute
    cols, rows, err = _execute(sql)
    if err:
        log.info("[report/generate] SQL execution error")
        resp = _err_resp("no_data", used_llm_fallback)
        resp["generated_sql"] = sql
        return JSONResponse(content=resp)
    if not rows:
        log.info("[report/generate] query returned 0 rows")
        resp = _err_resp("no_data", used_llm_fallback)
        resp["generated_sql"] = sql
        return JSONResponse(content=resp)

    log.info("[report/generate] done — %d rows", len(rows))
    return ReportResponse(
        title=f"Report: {query[:100]}",
        sections=[ReportSection(title="Query Results", columns=cols, rows=rows, sql=sql)],
        generated_sql=sql,
        provider=provider_used,
        used_llm_fallback=used_llm_fallback,
    )


@router.post("/run-saved", response_model=ReportResponse)
def run_saved_template(req: RunSavedRequest) -> ReportResponse:
    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="sql must not be empty")

    log.info('[report/run-saved] name="%s"', req.name)

    ok, reason = validate(sql)
    if not ok:
        return ReportResponse(
            title="Template Execution Failed",
            sections=[ReportSection(title="Error", error=f"SQL validation: {reason}")],
        )

    params: dict | None = None
    if req.start_date or req.end_date:
        params = {}
        if req.start_date:
            params["start_date"] = req.start_date
        if req.end_date:
            params["end_date"] = req.end_date

    cols, rows, err = _execute(sql, params)
    section = ReportSection(
        title=req.name,
        columns=cols,
        rows=rows,
        sql=sql,
        error=err,
    )
    log.info("[report/run-saved] done — %d rows, error=%s", len(rows), err)

    return ReportResponse(
        title=f"Saved Report: {req.name}",
        sections=[section],
    )


# ── Template management ────────────────────────────────────────────────────

@router.get("/saved-templates")
def list_saved_templates() -> list[dict]:
    return get_all_templates()


@router.post("/normalize-template")
def normalize_and_save_template(req: NormalizeTemplateRequest) -> dict:
    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="sql must not be empty")

    ok, reason = validate(sql)
    if not ok:
        raise HTTPException(status_code=400, detail=f"SQL validation: {reason}")

    table_names = _extract_table_names(sql)
    try:
        ddl_context = compact_ddl_for(table_names) if table_names else ""
    except Exception:
        ddl_context = ""

    try:
        norm = call_normalization_llm(sql, ddl_context)
    except Exception as exc:
        log.warning("[report/normalize-template] normalization failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Normalization failed: {exc}")

    tid = save_template(
        name=req.name,
        original_prompt=req.query,
        original_sql=sql,
        template_sql=normalize_saved_query(norm["template_sql"]),
        has_granularity=bool(norm.get("has_granularity", False)),
        inferred_date_column=str(norm.get("inferred_date_column", "")),
        date_injection_needed=bool(norm.get("date_injection_needed", False)),
        columns=[],
    )
    log.info("[report/normalize-template] saved id=%s name=%r", tid, req.name)
    return get_template(tid)


@router.patch("/saved-templates/{template_id}")
def rename_saved_template(template_id: str, req: RenameTemplateRequest) -> dict:
    if not get_template(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")
    rename_template(template_id, name)
    return get_template(template_id)


@router.delete("/saved-templates/{template_id}")
def delete_saved_template(template_id: str) -> dict:
    if not get_template(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    delete_template(template_id)
    return {"ok": True}


@router.post("/run-template", response_model=ReportResponse)
def run_template_by_id(req: RunTemplateRequest) -> ReportResponse:
    tpl = get_template(req.saved_query_id)
    if not tpl:
        raise HTTPException(status_code=404, detail="Saved query not found")

    sql = (tpl.get("template_sql") or tpl["original_sql"]).strip()

    sql = sql.replace("{{DATE_FROM}}", req.date_from)
    sql = sql.replace("{{DATE_TO}}", req.date_to)
    sql = sql.replace("{{start_date}}", req.date_from)
    sql = sql.replace("{{end_date}}", req.date_to)
    sql = sql.replace("{{GRANULARITY}}", req.granularity or "month")
    # Heal templates where the LLM forgot to quote the granularity inside DATE_TRUNC
    sql = re.sub(
        r"\bDATE_TRUNC\s*\(\s*(day|week|month)\s*,",
        lambda m: f"DATE_TRUNC('{m.group(1)}',",
        sql,
        flags=re.IGNORECASE,
    )

    log.info(
        '[report/run-template] id=%s name="%s" date_from=%s date_to=%s gran=%s',
        req.saved_query_id, tpl["name"], req.date_from, req.date_to, req.granularity,
    )

    ok, reason = validate(sql)
    if not ok:
        return ReportResponse(
            title="Template Execution Failed",
            sections=[ReportSection(title="Error", error=f"SQL validation: {reason}")],
        )

    cols, rows, err = _execute(sql)
    section = ReportSection(
        title=tpl["name"],
        columns=cols,
        rows=rows,
        sql=sql,
        error=err,
    )
    log.info("[report/run-template] done — %d rows, error=%s", len(rows), err)

    return ReportResponse(
        title=f"Saved Report: {tpl['name']}",
        sections=[section],
    )


@router.post("/migrate-templates")
def migrate_templates(req: MigrateTemplatesRequest) -> dict:
    """Accept a batch of localStorage-format templates, normalize and persist them."""
    existing_sqls = {t["original_sql"] for t in get_all_templates()}
    migrated = 0

    for item in req.templates:
        sql = (item.get("sql") or "").strip()
        if not sql:
            continue
        if sql in existing_sqls:
            continue

        ok, _ = validate(sql)
        if not ok:
            log.warning("[migrate-templates] skipping invalid SQL for %r", item.get("name"))
            continue

        table_names = _extract_table_names(sql)
        try:
            ddl_context = compact_ddl_for(table_names) if table_names else ""
        except Exception:
            ddl_context = ""

        try:
            norm = call_normalization_llm(sql, ddl_context)
            template_sql        = normalize_saved_query(norm["template_sql"])
            has_granularity     = bool(norm.get("has_granularity", False))
            inferred_date_col   = str(norm.get("inferred_date_column", ""))
            date_injection      = bool(norm.get("date_injection_needed", False))
        except Exception as exc:
            log.warning("[migrate-templates] normalization failed for %r: %s", item.get("name"), exc)
            # Graceful fallback: promote :start_date/:end_date params to {{}} placeholders
            template_sql = re.sub(r":start_date\b", "{{DATE_FROM}}", sql, flags=re.IGNORECASE)
            template_sql = re.sub(r":end_date\b",   "{{DATE_TO}}",   template_sql, flags=re.IGNORECASE)
            template_sql = normalize_saved_query(template_sql)
            has_granularity   = False
            inferred_date_col = ""
            date_injection    = not bool(re.search(r":start_date|:end_date", sql, re.IGNORECASE))

        save_template(
            name=item.get("name") or "Migrated Template",
            original_prompt=item.get("query") or "",
            original_sql=sql,
            template_sql=template_sql,
            has_granularity=has_granularity,
            inferred_date_column=inferred_date_col,
            date_injection_needed=date_injection,
            columns=[],
        )
        existing_sqls.add(sql)
        migrated += 1
        log.info("[migrate-templates] migrated %r", item.get("name"))

    return {"migrated": migrated, "templates": get_all_templates()}
