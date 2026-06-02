"""
Deterministic SQL normalization applied to a template before it is persisted.

Two responsibilities:
  1. Replace hardcoded ISO date literals that the LLM normalizer missed with
     {{start_date}} / {{end_date}} placeholders.
  2. Rewrite the GROUP BY clause when it doesn't match the non-aggregated
     expressions in the SELECT clause.
"""
import re

# ── Aggregate-function detection ───────────────────────────────────────────

_AGG_START_RE = re.compile(
    r"^\s*(?:SUM|COUNT|AVG|MIN|MAX|STDDEV|VARIANCE|ARRAY_AGG|STRING_AGG"
    r"|BOOL_AND|BOOL_OR|JSON_AGG|JSONB_AGG|BIT_AND|BIT_OR|EVERY"
    r"|PERCENTILE_CONT|PERCENTILE_DISC)\s*\(",
    re.IGNORECASE,
)
_COUNT_DISTINCT_RE = re.compile(r"^\s*COUNT\s*\(\s*DISTINCT\b", re.IGNORECASE)


def _is_aggregate(expr: str) -> bool:
    return bool(_AGG_START_RE.match(expr)) or bool(_COUNT_DISTINCT_RE.match(expr))


# ── Comma-split that respects parenthesis depth ────────────────────────────

def _split_by_comma(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


# ── SELECT non-aggregate column extraction ─────────────────────────────────

_SELECT_FROM_RE = re.compile(
    r"\bSELECT\b(.*?)\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)
_ALIAS_RE = re.compile(r"\s+AS\s+\w+\s*$", re.IGNORECASE)


def _non_agg_select_cols(sql: str) -> list[str]:
    """Return the non-aggregated SELECT expressions with aliases stripped."""
    m = _SELECT_FROM_RE.search(sql)
    if not m:
        return []
    items = _split_by_comma(m.group(1).strip())
    result: list[str] = []
    for item in items:
        clean = _ALIAS_RE.sub("", item).strip()
        if not clean or clean == "*":
            return []  # SELECT * — can't determine non-agg columns safely
        if not _is_aggregate(clean):
            result.append(clean)
    return result


# ── GROUP BY correction ────────────────────────────────────────────────────

_GROUP_BY_RE = re.compile(
    r"\bGROUP\s+BY\b(.*?)"
    r"(?=\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\bUNION\b|\bEXCEPT\b|\bINTERSECT\b|;|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", " ", expr.strip()).upper()


def _fix_group_by(sql: str) -> str:
    """Rewrite GROUP BY when it doesn't match the non-aggregated SELECT columns."""
    m = _GROUP_BY_RE.search(sql)
    if not m:
        return sql

    non_agg = _non_agg_select_cols(sql)
    if not non_agg:
        return sql

    current_gb_cols = _split_by_comma(m.group(1).strip())
    if not current_gb_cols:
        return sql

    if {_norm_expr(c) for c in non_agg} == {_norm_expr(c) for c in current_gb_cols}:
        return sql  # already correct

    correct_gb = ", ".join(non_agg)
    return sql[: m.start()] + f"GROUP BY {correct_gb}" + sql[m.end() :]


# ── Date literal replacement ───────────────────────────────────────────────

_ISO_DATE = r"'(\d{4}-\d{2}-\d{2})'"

# BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
_BETWEEN_RE = re.compile(
    rf"BETWEEN\s+{_ISO_DATE}\s+AND\s+{_ISO_DATE}",
    re.IGNORECASE,
)
# Two-char operators (must be processed before single-char to avoid partial match)
_GTE_RE = re.compile(rf"(>=\s*){_ISO_DATE}", re.IGNORECASE)
_LTE_RE = re.compile(rf"(<=\s*){_ISO_DATE}", re.IGNORECASE)
# Single-char operators with negative lookahead to avoid matching >= / <=
_GT_RE = re.compile(rf"(>(?!=)\s*){_ISO_DATE}", re.IGNORECASE)
_LT_RE = re.compile(rf"(<(?![=>])\s*){_ISO_DATE}", re.IGNORECASE)

# EXTRACT(YEAR FROM col) = YYYY
_EXTRACT_YEAR_RE = re.compile(
    r"EXTRACT\s*\(\s*YEAR\s+FROM\s+(\w+)\s*\)\s*=\s*\d{4}",
    re.IGNORECASE,
)
# YEAR(col) = YYYY  (MySQL-style, occasionally generated)
_YEAR_FN_RE = re.compile(
    r"YEAR\s*\(\s*(\w+)\s*\)\s*=\s*\d{4}",
    re.IGNORECASE,
)
# TO_CHAR(col, 'YYYY-MM') with any comparison value
_TO_CHAR_RE = re.compile(
    r"TO_CHAR\s*\(\s*(\w+)\s*,\s*'YYYY-MM'\s*\)\s*"
    r"(?:BETWEEN\s*'[\d-]+'\s*AND\s*'[\d-]+'|[<>=!]+\s*'[\d-]+')",
    re.IGNORECASE,
)


def _replace_date_literals(sql: str) -> str:
    """Replace hardcoded ISO date literals with {{start_date}} / {{end_date}}."""
    # BETWEEN covers both dates in one substitution — run first
    sql = _BETWEEN_RE.sub("BETWEEN '{{start_date}}' AND '{{end_date}}'", sql)
    # Two-char operators before single-char
    sql = _GTE_RE.sub(r"\1'{{start_date}}'", sql)
    sql = _LTE_RE.sub(r"\1'{{end_date}}'", sql)
    sql = _GT_RE.sub(r"\1'{{start_date}}'", sql)
    sql = _LT_RE.sub(r"\1'{{end_date}}'", sql)
    # EXTRACT(YEAR FROM col) = YYYY → range filter on the column
    sql = _EXTRACT_YEAR_RE.sub(
        r"\1 >= '{{start_date}}' AND \1 <= '{{end_date}}'", sql
    )
    # YEAR(col) = YYYY → range filter on the column
    sql = _YEAR_FN_RE.sub(
        r"\1 >= '{{start_date}}' AND \1 <= '{{end_date}}'", sql
    )
    # TO_CHAR(col, 'YYYY-MM') comparisons → range filter on the column
    sql = _TO_CHAR_RE.sub(
        r"\1 >= '{{start_date}}' AND \1 <= '{{end_date}}'", sql
    )
    return sql


# ── Public API ─────────────────────────────────────────────────────────────

def normalize_saved_query(sql: str) -> str:
    """Normalize a SQL template before it is persisted.

    Runs after the LLM normalizer as a deterministic safety net:
    1. Replace any remaining hardcoded date literals with
       {{start_date}} / {{end_date}}.
    2. Fix GROUP BY to match the non-aggregated SELECT columns.
    """
    sql = _replace_date_literals(sql)
    sql = _fix_group_by(sql)
    return sql
