"""
EPMS AI Dashboard Server  —  port 8001
Run with:   python dashboard_server.py
Or:         uvicorn dashboard_server:app --port 8001 --reload

The original text-to-SQL server (main.py) stays on port 8000.
This server shares the same database, retrieval indexes, and LLM modules
but runs as a completely separate process.
"""

import decimal
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import engine
import metadata_loader
from metadata_loader import load_all_cards, compact_ddl_for
from retrieval import initialize as init_retrieval, retrieve_tables
from sql_validator import validate, ensure_limit
from layout_schema import DashboardLayout, WidgetConfig
from layout_prompt import build_layout_messages
from llm import call_llm, get_provider, list_providers, PROVIDER_LABELS
from prompt_builder import strip_fences
from report_router import router as report_router
from chat_router import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("epms-dashboard")

DASHBOARD_PORT    = 8001
STATIC_DIR        = Path(__file__).parent / "dashboard_static"
REPORT_STATIC_DIR = Path(__file__).parent / "report_static"
CHAT_STATIC_DIR   = Path(__file__).parent / "chat_static"

app = FastAPI(
    title="EPMS AI Dashboard",
    description="Natural-language dashboard generator connected to the EPMS database.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/report-static", StaticFiles(directory=str(REPORT_STATIC_DIR)), name="report-static")
app.mount("/chat-static", StaticFiles(directory=str(CHAT_STATIC_DIR)), name="chat-static")

app.include_router(report_router)
app.include_router(chat_router)


# ── startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    cards = load_all_cards()
    init_retrieval(cards)
    log.info(
        "EPMS AI Dashboard ready — http://127.0.0.1:%d  (%d tables indexed)",
        DASHBOARD_PORT, len(cards),
    )


# ── request / response models ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    goal: str
    max_widgets: int = 5


class WidgetData(BaseModel):
    columns: list[str] = []
    rows: list[dict] = []
    error: Optional[str] = None


class DashboardResponse(BaseModel):
    title: str
    widgets: list[WidgetConfig]
    tables_used: list[str]
    widget_data: dict[str, WidgetData]
    provider: str
    provider_label: str
    tables_considered: list[str]
    used_llm_fallback: bool = False


# ── helpers ────────────────────────────────────────────────────────────────

def _coerce(v):
    """Convert DB-native types that are not JSON-serialisable."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return str(v)
    return v


def _execute_safe(
    sql: str, max_rows: int = 500
) -> tuple[list[str], list[dict], Optional[str]]:
    """Run a SELECT against the EPMS database. Returns (columns, rows, error_or_None)."""
    try:
        limited = ensure_limit(sql, max_rows)
        with engine.connect() as conn:
            result  = conn.execute(text(limited))
            columns = list(result.keys())
            rows    = [
                {k: _coerce(v) for k, v in dict(r._mapping).items()}
                for r in result.fetchall()
            ]
        return columns, rows, None
    except SQLAlchemyError as exc:
        err = str(exc.orig) if hasattr(exc, "orig") and exc.orig else str(exc)
        log.warning("[execute_safe] %s", err[:200])
        return [], [], err
    except Exception as exc:
        log.warning("[execute_safe] unexpected: %s", exc)
        return [], [], str(exc)


# ── routes ─────────────────────────────────────────────────────────────────

# Browsers heuristically cache FileResponse pages; force revalidation so UI
# changes show up without a hard refresh.
_NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE)


@app.get("/report")
def report_page():
    return FileResponse(REPORT_STATIC_DIR / "report.html", headers=_NO_CACHE)


@app.get("/chat")
def chat_page():
    return FileResponse(CHAT_STATIC_DIR / "chat.html", headers=_NO_CACHE)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": get_provider(),
        "providers": list_providers(),
    }


@app.post("/api/generate", response_model=DashboardResponse)
def generate_dashboard(req: GenerateRequest) -> DashboardResponse:
    goal        = (req.goal or "").strip()
    max_widgets = max(3, min(req.max_widgets, 12))

    if not goal:
        return JSONResponse(status_code=400, content={"detail": "goal must not be empty"})

    t0 = time.monotonic()
    log.info('[dashboard] goal="%s"', goal)

    # ── 1. Retrieve tables ────────────────────────────────────────────────
    table_cards, used_llm_fallback = retrieve_tables(goal, k=5)
    table_names = [c.name for c in table_cards]
    log.info("[dashboard] retrieved tables: %s (llm_fallback=%s)", table_names, used_llm_fallback)

    if not table_cards:
        from error_handler import error_response, is_off_domain
        kind = "off_domain" if is_off_domain(goal) else "no_data"
        log.info("[dashboard] no tables retrieved — returning %s", kind)
        return JSONResponse(content=error_response(kind, used_llm_fallback))

    # ── 2. Build compact DDL ──────────────────────────────────────────────
    ddl = compact_ddl_for(table_names)

    # ── 3. Build prompt → call LLM ───────────────────────────────────────
    messages = build_layout_messages(ddl, goal, max_widgets)
    raw_text, provider_used, usage = call_llm(messages, max_tokens=1000, temperature=0)
    log.info(
        "[dashboard] provider=%s input=%d output=%d",
        provider_used, usage["input_tokens"], usage["output_tokens"],
    )

    # ── 4. Parse layout JSON ──────────────────────────────────────────────
    clean_text = strip_fences(raw_text)
    try:
        layout = DashboardLayout.model_validate_json(clean_text)
    except ValidationError as exc:
        log.warning("[dashboard] layout parse failed: %s | raw=%r", exc, raw_text[:300])
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "raw_llm_output": raw_text},
        )

    # ── 5. Validate and execute each sql_hint ─────────────────────────────
    widget_data: dict[str, WidgetData] = {}
    for widget in layout.widgets:
        if not widget.sql_hint:
            widget_data[widget.id] = WidgetData(error="No SQL hint provided")
            continue

        ok, reason = validate(widget.sql_hint)
        if not ok:
            widget.sql_hint = None
            widget_data[widget.id] = WidgetData(error=f"SQL validation: {reason}")
            continue

        columns, rows, err = _execute_safe(widget.sql_hint)
        if err:
            widget_data[widget.id] = WidgetData(error=err)
        else:
            widget_data[widget.id] = WidgetData(columns=columns, rows=rows)

    elapsed = time.monotonic() - t0
    log.info("[dashboard] done %.2fs, widgets=%d", elapsed, len(layout.widgets))

    return DashboardResponse(
        title=layout.title,
        widgets=layout.widgets,
        tables_used=layout.tables_used,
        widget_data=widget_data,
        provider=provider_used,
        provider_label=PROVIDER_LABELS.get(provider_used, provider_used),
        tables_considered=table_names,
        used_llm_fallback=used_llm_fallback,
    )


# ── entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=DASHBOARD_PORT,
        reload=True,
    )
