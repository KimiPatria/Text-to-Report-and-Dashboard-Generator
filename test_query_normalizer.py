"""Unit tests for query_normalizer.normalize_saved_query."""
import re
import pytest
from query_normalizer import normalize_saved_query

# ── Date pattern replacement ───────────────────────────────────────────────

def test_between_replaces_both_dates():
    sql = (
        "SELECT x FROM t "
        "WHERE oph_approved_date BETWEEN '2026-03-31' AND '2026-05-30'"
    )
    result = normalize_saved_query(sql)
    assert "BETWEEN '{{start_date}}' AND '{{end_date}}'" in result
    assert not re.search(r"'\d{4}-\d{2}-\d{2}'", result)


def test_gte_lte_replaces_start_and_end():
    sql = (
        "SELECT x FROM t "
        "WHERE date_col >= '2026-01-01' AND date_col <= '2026-12-31'"
    )
    result = normalize_saved_query(sql)
    assert ">= '{{start_date}}'" in result
    assert "<= '{{end_date}}'" in result
    assert not re.search(r"'\d{4}-\d{2}-\d{2}'", result)


def test_gte_lt_replaces_start_and_end():
    sql = (
        "SELECT x FROM t "
        "WHERE date_col >= '2026-01-01' AND date_col < '2027-01-01'"
    )
    result = normalize_saved_query(sql)
    assert ">= '{{start_date}}'" in result
    assert "< '{{end_date}}'" in result
    assert not re.search(r"'\d{4}-\d{2}-\d{2}'", result)


def test_extract_year_replaced_with_range():
    sql = "SELECT x FROM t WHERE EXTRACT(YEAR FROM date_col) = 2026"
    result = normalize_saved_query(sql)
    assert "date_col >= '{{start_date}}'" in result
    assert "date_col <= '{{end_date}}'" in result


def test_year_function_replaced_with_range():
    sql = "SELECT x FROM t WHERE YEAR(created_at) = 2025"
    result = normalize_saved_query(sql)
    assert "created_at >= '{{start_date}}'" in result
    assert "created_at <= '{{end_date}}'" in result


def test_to_char_comparison_replaced():
    sql = "SELECT x FROM t WHERE TO_CHAR(txn_date, 'YYYY-MM') = '2026-03'"
    result = normalize_saved_query(sql)
    assert "txn_date >= '{{start_date}}'" in result
    assert "txn_date <= '{{end_date}}'" in result


def test_existing_llm_placeholders_preserved():
    # LLM already normalised with {{DATE_FROM}} / {{DATE_TO}} — must not be touched
    sql = (
        "SELECT x FROM t "
        "WHERE date_col BETWEEN '{{DATE_FROM}}' AND '{{DATE_TO}}'"
    )
    result = normalize_saved_query(sql)
    assert "{{DATE_FROM}}" in result
    assert "{{DATE_TO}}" in result
    assert not re.search(r"'\d{4}-\d{2}-\d{2}'", result)


def test_no_dates_passthrough_unchanged():
    sql = "SELECT x, COUNT(*) AS cnt FROM t GROUP BY x"
    assert normalize_saved_query(sql) == sql


# ── GROUP BY mismatch correction ───────────────────────────────────────────

def test_group_by_mismatch_corrected():
    sql = (
        "SELECT oph_division_code, SUM(bunches_total) AS total_bunches "
        "FROM t_oph "
        "WHERE oph_approved_date BETWEEN '2026-03-31' AND '2026-05-30' "
        "GROUP BY DATE_TRUNC('day', oph_approved_date)"
    )
    result = normalize_saved_query(sql)
    # Date placeholders applied
    assert "BETWEEN '{{start_date}}' AND '{{end_date}}'" in result
    # GROUP BY corrected to match the only non-aggregate SELECT column
    assert "GROUP BY oph_division_code" in result
    assert "DATE_TRUNC" not in result.split("GROUP BY")[1]


def test_group_by_correct_not_changed():
    sql = (
        "SELECT division, SUM(amount) AS total "
        "FROM sales "
        "GROUP BY division"
    )
    result = normalize_saved_query(sql)
    assert "GROUP BY division" in result


def test_group_by_multiple_non_agg_columns():
    sql = (
        "SELECT region, division, SUM(qty) AS total_qty "
        "FROM t "
        "GROUP BY region"  # missing division
    )
    result = normalize_saved_query(sql)
    # Both non-agg columns should appear in GROUP BY
    gb_part = result.split("GROUP BY")[1]
    assert "region" in gb_part
    assert "division" in gb_part


def test_group_by_with_granularity_placeholder():
    sql = (
        "SELECT DATE_TRUNC('{{GRANULARITY}}', approved_date) AS period, "
        "SUM(bunches_total) AS total "
        "FROM t_oph "
        "GROUP BY DATE_TRUNC('day', approved_date)"
    )
    result = normalize_saved_query(sql)
    gb_part = result.split("GROUP BY")[1]
    assert "{{GRANULARITY}}" in gb_part
    assert "'day'" not in gb_part


def test_query_with_no_group_by_unchanged():
    sql = "SELECT x FROM t WHERE x = 1"
    assert normalize_saved_query(sql) == sql
