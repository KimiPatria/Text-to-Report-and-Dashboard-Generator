import re
import sqlparse
from sqlparse.tokens import Keyword, DML, CTE

DENY_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "MERGE", "CALL", "COPY", "EXEC", "EXECUTE",
    "VACUUM", "ANALYZE", "REINDEX", "CLUSTER", "COMMENT", "SECURITY",
    "SET", "RESET", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "LISTEN", "NOTIFY", "LOCK", "DECLARE", "FETCH", "MOVE", "PREPARE",
    "DEALLOCATE", "DO",
}

DANGEROUS_FUNCS = re.compile(
    r"\b(pg_sleep|pg_read_file|pg_read_binary_file|lo_import|lo_export|"
    r"dblink|pg_terminate_backend|pg_cancel_backend|copy_from|copy_to)\b",
    re.IGNORECASE,
)


def _first_meaningful_token(stmt) -> str | None:
    for tok in stmt.flatten():
        if tok.is_whitespace:
            continue
        if tok.ttype in (sqlparse.tokens.Comment, sqlparse.tokens.Comment.Single,
                         sqlparse.tokens.Comment.Multiline):
            continue
        return tok.normalized.upper()
    return None


def validate(sql: str) -> tuple[bool, str | None]:
    if not sql or not sql.strip():
        return False, "Empty SQL."

    cleaned = sqlparse.format(sql, strip_comments=True).strip().rstrip(";").strip()
    if not cleaned:
        return False, "SQL contained only comments."

    if DANGEROUS_FUNCS.search(cleaned):
        return False, "SQL references a disallowed function."

    statements = [s for s in sqlparse.parse(cleaned) if str(s).strip()]
    if len(statements) != 1:
        return False, "Only a single SQL statement is allowed."

    stmt = statements[0]
    first = _first_meaningful_token(stmt)
    if first not in {"SELECT", "WITH"}:
        return False, f"Only SELECT or WITH queries are allowed (got: {first})."

    for tok in stmt.flatten():
        if tok.ttype in (Keyword, DML, CTE):
            word = tok.normalized.upper()
            if word in DENY_KEYWORDS:
                return False, f"Disallowed keyword: {word}."

    return True, None


_LIMIT_RE = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)


def ensure_limit(sql: str, max_rows: int) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if _LIMIT_RE.search(cleaned):
        return cleaned
    return f"{cleaned} LIMIT {max_rows}"
