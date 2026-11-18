"""Guardrails for LLM-generated SQL.

Never run model output against a database unchecked. This validator enforces:
read-only single statements, an allowlist of tables, no dangerous keywords,
and a row LIMIT. The connection itself is also opened read-only, so the guard
is defense-in-depth rather than the only line of defense.
"""
from __future__ import annotations

import re

ALLOWED_TABLES = {"stores", "products", "inventory", "sales"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|grant|truncate)\b",
    re.I,
)
MAX_ROWS = 200


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    cleaned = re.sub(r"^\s*```(?:sql)?|```\s*$", "", sql, flags=re.I)  # LLMs often wrap SQL in fences
    cleaned = re.sub(r"--[^\n]*|/\*.*?\*/", " ", cleaned, flags=re.S).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("empty query")
    if ";" in cleaned:
        raise UnsafeSQLError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise UnsafeSQLError("only SELECT queries are allowed")
    if FORBIDDEN.search(cleaned):
        raise UnsafeSQLError("query contains a forbidden keyword")
    referenced = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", cleaned, re.I)}
    ctes = {t.lower() for t in re.findall(r"\b([a-zA-Z_]\w*)\s+as\s*\(", cleaned, re.I)}
    unknown = referenced - ALLOWED_TABLES - ctes
    if unknown:
        raise UnsafeSQLError(f"table(s) not allowed: {sorted(unknown)}")
    if not re.search(r"\blimit\s+\d+\s*$", cleaned, re.I):
        cleaned = f"{cleaned} LIMIT {MAX_ROWS}"
    return cleaned
