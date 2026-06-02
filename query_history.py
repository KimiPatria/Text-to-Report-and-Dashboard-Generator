import json
import logging
import sqlite3
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
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                provider TEXT NOT NULL,
                tables_used TEXT,
                success INTEGER NOT NULL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_success ON queries(success)")
        conn.commit()
    log.info("query_history.db initialized at %s", _DB_PATH.resolve())


def record_query(
    question: str,
    sql: str,
    provider: str,
    tables_used: list[str],
    success: bool,
    error: str | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO queries (question, sql, provider, tables_used, success, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                sql,
                provider,
                json.dumps(tables_used) if tables_used else None,
                1 if success else 0,
                error,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
    log.debug("Recorded query id=%d success=%s", row_id, success)
    return row_id


def get_successful_pairs(limit: int = 500) -> list[tuple[int, str, str]]:
    """Return (id, question, sql) for successful queries, most recent first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, sql FROM queries WHERE success = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [(r["id"], r["question"], r["sql"]) for r in rows]
