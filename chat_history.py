"""Session-scoped chat history store for the EPMS chatbot.

Persists full conversation turns (question, answer, optional SQL + data rows)
per session in the same SQLite file as query_history.py. Rows are capped at
write time so the store — and any prompt built from it — stays bounded.
"""

import json
import logging
import uuid

from query_history import _connect

log = logging.getLogger(__name__)

# Max data rows persisted per turn. Anything beyond this was never shown to
# the LLM anyway (answer prompt previews 50), so storing more is waste.
MAX_STORED_ROWS = 50

_TITLE_LEN = 60


def init_chat_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sql TEXT,
                columns TEXT,
                rows TEXT,
                row_count INTEGER DEFAULT 0,
                stats TEXT,
                route TEXT DEFAULT 'sql',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON chat_turns(session_id, id)"
        )
        conn.commit()
    log.info("chat history tables ready")


def ensure_session(session_id: str | None, first_question: str) -> str:
    """Return a valid session id, creating the session row on first use."""
    sid = (session_id or "").strip() or str(uuid.uuid4())
    title = first_question.strip()[:_TITLE_LEN]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP",
            (sid, title or "New chat"),
        )
        conn.commit()
    return sid


def add_turn(
    session_id: str,
    question: str,
    answer: str,
    sql: str | None = None,
    columns: list[str] | None = None,
    rows: list[dict] | None = None,
    row_count: int = 0,
    stats: dict | None = None,
    route: str = "sql",
) -> None:
    capped = rows[:MAX_STORED_ROWS] if rows else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_turns
                (session_id, question, answer, sql, columns, rows, row_count, stats, route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                sql,
                json.dumps(columns, default=str) if columns else None,
                json.dumps(capped, default=str) if capped else None,
                row_count,
                json.dumps(stats, default=str) if stats else None,
                route,
            ),
        )
        conn.commit()


def _parse_turn(r) -> dict:
    return {
        "question": r["question"],
        "answer": r["answer"],
        "sql": r["sql"],
        "columns": json.loads(r["columns"]) if r["columns"] else [],
        "rows": json.loads(r["rows"]) if r["rows"] else [],
        "row_count": r["row_count"] or 0,
        "stats": json.loads(r["stats"]) if r["stats"] else {},
        "route": r["route"] or "sql",
        "created_at": r["created_at"],
    }


def get_turns(session_id: str, limit: int = 20) -> list[dict]:
    """Most recent turns of a session, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM (SELECT * FROM chat_turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (session_id, limit),
        ).fetchall()
    return [_parse_turn(r) for r in rows]


def list_sessions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(t.id) AS turn_count
            FROM chat_sessions s
            LEFT JOIN chat_turns t ON t.session_id = s.id
            GROUP BY s.id
            HAVING turn_count > 0
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        conn.commit()
    return cur.rowcount > 0


# Tables must exist before the first request hits the router.
init_chat_db()
