"""
Persistent storage for saved SQL report templates.
Uses the same query_history.db SQLite file as query_history.py.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_DB_PATH = Path("./query_history.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_templates (
                id                   TEXT PRIMARY KEY,
                name                 TEXT NOT NULL,
                original_prompt      TEXT NOT NULL DEFAULT '',
                original_sql         TEXT NOT NULL,
                template_sql         TEXT,
                has_granularity      INTEGER NOT NULL DEFAULT 0,
                inferred_date_column TEXT NOT NULL DEFAULT '',
                date_injection_needed INTEGER NOT NULL DEFAULT 0,
                columns              TEXT NOT NULL DEFAULT '[]',
                created_at           TEXT NOT NULL
            )
        """)
        conn.commit()
    log.info("saved_templates table ready in %s", _DB_PATH.resolve())


def save_template(
    *,
    name: str,
    original_prompt: str,
    original_sql: str,
    template_sql: str,
    has_granularity: bool,
    inferred_date_column: str,
    date_injection_needed: bool,
    columns: list[str],
) -> str:
    tid = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO saved_templates
              (id, name, original_prompt, original_sql, template_sql,
               has_granularity, inferred_date_column, date_injection_needed,
               columns, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                tid,
                name,
                original_prompt,
                original_sql,
                template_sql,
                int(has_granularity),
                inferred_date_column,
                int(date_injection_needed),
                json.dumps(columns),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    log.info("Saved template id=%s name=%r", tid, name)
    return tid


def get_all_templates() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM saved_templates ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_template(tid: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM saved_templates WHERE id = ?", (tid,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def update_template_normalization(
    tid: str,
    *,
    template_sql: str,
    has_granularity: bool,
    inferred_date_column: str,
    date_injection_needed: bool,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE saved_templates
               SET template_sql = ?, has_granularity = ?,
                   inferred_date_column = ?, date_injection_needed = ?
             WHERE id = ?
            """,
            (
                template_sql,
                int(has_granularity),
                inferred_date_column,
                int(date_injection_needed),
                tid,
            ),
        )
        conn.commit()


def rename_template(tid: str, name: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE saved_templates SET name = ? WHERE id = ?", (name, tid))
        conn.commit()


def delete_template(tid: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM saved_templates WHERE id = ?", (tid,))
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["has_granularity"] = bool(d["has_granularity"])
    d["date_injection_needed"] = bool(d["date_injection_needed"])
    try:
        d["columns"] = json.loads(d["columns"])
    except Exception:
        d["columns"] = []
    return d
