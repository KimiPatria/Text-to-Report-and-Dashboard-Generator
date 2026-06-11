import logging

log = logging.getLogger(__name__)

_LAYOUT_SYSTEM = """\
You are a dashboard layout expert for an EPMS palm-oil plantation analytics system.
Given table DDL and a dashboard goal, return a JSON dashboard layout — NO markdown fences, NO preamble, valid JSON only.

Output schema (exact shape required):
{
  "title": "<dashboard title>",
  "widgets": [
    {
      "id": "w1",
      "chart_type": "<bar|line|pie|big_number|table|scatter>",
      "title": "<widget title>",
      "x_field": "<column name or null>",
      "y_field": "<column name or null>",
      "group_by": "<column name or null>",
      "table": "<table name>",
      "sql_hint": "<SELECT skeleton using only columns from DDL, or null>",
      "grid": {"x": 0, "y": 0, "w": 6, "h": 4}
    }
  ],
  "tables_used": ["<table1>", "<table2>"]
}

Rules:
- chart_type must be one of: bar, line, pie, big_number, table, scatter
- Use big_number for single KPI metrics (one aggregated value)
- Grid uses 12-column layout: w=6 means half-width, w=12 means full-width, w=4 means one-third
- Place widgets in a logical reading order; avoid overlapping grid cells
- Include between 3 and MAX_WIDGETS widgets (replace MAX_WIDGETS with the number in the request)
- sql_hint must be a valid SELECT skeleton referencing only columns present in the DDL; set to null if unsure
- Use only tables and columns present in the DDL block
- tables_used must list every table referenced by at least one widget
- NEVER use ROLLUP, GROUPING SETS, CUBE, or UNION ALL to add subtotal/grand-total rows in any sql_hint — chart widgets must return only atomic data rows, never summary rows
- For time-series or grouped charts the sql_hint must filter to individual periods (e.g. GROUP BY month, GROUP BY estate) with no total/rollup rows; grand totals belong in a separate big_number widget
"""


def build_layout_messages(ddl: str, goal: str, max_widgets: int) -> list[dict]:
    """Build chat messages for the dashboard layout suggester.

    Returns a two-element list [system, user] matching the same dict structure
    used by build_sql_messages() in prompt_builder.py.
    """
    system_content = _LAYOUT_SYSTEM.replace("MAX_WIDGETS", str(max_widgets))

    user_content = (
        f"DDL:\n{ddl}\n\n"
        f"Dashboard goal: {goal}\n\n"
        f"Return a JSON layout with 3–{max_widgets} widgets."
    )

    log.info(
        "[layout_prompt] ddl_chars=%d goal_chars=%d max_widgets=%d",
        len(ddl), len(goal), max_widgets,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
