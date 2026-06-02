#!/usr/bin/env python3
"""Dump DB schema + existing metadata to a text file for pasting into Claude.ai."""
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from config import engine, DB_SCHEMA

METADATA_PATH = Path("./schema_metadata.json")
OUTPUT_PATH   = Path("./schema_dump_for_claude.txt")


def main():
    col_sql = text("""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
    """)
    fk_sql = text("""
        SELECT kcu.table_name, kcu.column_name,
               ccu.table_name AS ref_table, ccu.column_name AS ref_col
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = :schema
    """)

    with engine.connect() as conn:
        col_rows = conn.execute(col_sql, {"schema": DB_SCHEMA}).fetchall()
        fk_rows  = conn.execute(fk_sql,  {"schema": DB_SCHEMA}).fetchall()

    cols_by_table: dict = {}
    for r in col_rows:
        cols_by_table.setdefault(r.table_name, []).append(
            f"  {r.column_name} ({r.data_type}, {'nullable' if r.is_nullable == 'YES' else 'not null'})"
        )

    fks_by_table: dict = {}
    for r in fk_rows:
        fks_by_table.setdefault(r.table_name, []).append(
            f"  {r.column_name} -> {r.ref_table}.{r.ref_col}"
        )

    # Load existing metadata to preserve manually_edited entries
    existing: dict = {}
    if METADATA_PATH.exists():
        raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        existing = raw.get("tables", raw)

    lines = []
    lines.append(f"SCHEMA: {DB_SCHEMA}")
    lines.append(f"TOTAL TABLES: {len(cols_by_table)}")
    lines.append("")

    for table in sorted(cols_by_table):
        meta = existing.get(table, {})
        manually_edited = meta.get("manually_edited", False)
        lines.append(f"TABLE: {table}" + (" [manually_edited — preserve as-is]" if manually_edited else ""))
        lines.extend(cols_by_table[table])
        if table in fks_by_table:
            lines.append("  FOREIGN KEYS:")
            lines.extend(f"  {fk}" for fk in fks_by_table[table])
        if meta and not manually_edited:
            lines.append("  EXISTING METADATA (preserve description/synonyms/example_questions if good):")
            lines.append(f"  description: {meta.get('description', '')}")
            syns = meta.get('synonyms', [])
            if syns:
                lines.append(f"  synonyms: {syns}")
            eqs = meta.get('example_questions', [])
            if eqs:
                lines.append(f"  example_questions: {eqs}")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cols_by_table)} tables to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"\nNext: Copy the contents of {OUTPUT_PATH} into Claude.ai with the prompt from the README.")


if __name__ == "__main__":
    main()
