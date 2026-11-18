"""Data agent: natural-language questions -> safe SQL -> answer.

LLM mode: the model writes SQLite from the schema; the query passes through
the SQL guard, runs on a read-only connection, and on an error the agent gets
one self-correction attempt with the error message (a common agentic pattern).

Offline mode: a small intent parser maps common analyst questions to
parameterized SQL templates. It covers the eval set and shows the contract the
LLM path must satisfy.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..llm import LLMClient
from ..observability.tracing import Tracer
from .sql_guard import UnsafeSQLError, validate_sql

SCHEMA_DOC = """SQLite tables:
stores(store_id INTEGER, name TEXT, city TEXT, state TEXT)
products(product_id INTEGER, name TEXT, category TEXT, unit_price REAL, unit_cost REAL)
inventory(store_id INTEGER, product_id INTEGER, on_hand INTEGER, reorder_point INTEGER)
sales(sale_id INTEGER, sale_date TEXT 'YYYY-MM-DD', store_id INTEGER, product_id INTEGER, quantity INTEGER, revenue REAL)
Data covers 2026-01-01 to 2026-03-31. Cities: Edison, Boston, Charlotte, Cincinnati, Santa Clara.
A product needs reorder when on_hand <= reorder_point."""

SYSTEM = (
    "You write a single SQLite SELECT query that answers the user's question.\n"
    f"{SCHEMA_DOC}\nRound money to 2 decimals. Return only SQL, no explanation."
)

MONTHS = {"january": "2026-01", "february": "2026-02", "march": "2026-03", "jan": "2026-01", "feb": "2026-02", "mar": "2026-03"}
CATEGORIES = ["Paper", "Ink & Toner", "Writing", "Technology", "Furniture", "Shipping"]


@dataclass
class DataAnswer:
    answer: str
    sql: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: Optional[str] = None


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


def format_table(columns: list[str], rows: list[tuple], max_rows: int = 10) -> str:
    if not rows:
        return "No matching records."
    fmt = lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows")
    return "\n".join(lines)


class DataAgent:
    def __init__(self, db_path: Path, llm: Optional[LLMClient], tracer: Tracer):
        self.con = connect_readonly(db_path)
        self.llm, self.tracer = llm, tracer
        self.cities = [r[0] for r in self.con.execute("SELECT city FROM stores")]
        self.products = [r[0] for r in self.con.execute("SELECT name FROM products")]

    # ---- execution -------------------------------------------------------
    def run_sql(self, sql: str, params: tuple = ()) -> tuple[list[str], list[tuple]]:
        with self.tracer.span("sql_execute", sql=sql) as span:
            cur = self.con.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            span.attrs["row_count"] = len(rows)
            return cols, rows

    # ---- offline intent parser ------------------------------------------
    def _filters(self, q: str) -> tuple[list[str], list[Any], str]:
        where, params, label = [], [], []
        city = next((c for c in self.cities if c.lower() in q), None)
        if city:
            where.append("s.city = ?"); params.append(city); label.append(city)
        month = next((v for k, v in MONTHS.items() if re.search(rf"\b{k}\b", q)), None)
        if month:
            where.append("substr(x.sale_date, 1, 7) = ?"); params.append(month); label.append(month)
        cat = next((c for c in CATEGORIES if c.lower() in q or c.lower().split()[0] in q.split()), None)
        if cat and "by category" not in q:
            where.append("p.category = ?"); params.append(cat); label.append(cat)
        return where, params, ", ".join(label)

    def template_sql(self, question: str) -> Optional[tuple[str, tuple]]:
        q = question.lower()
        where, params, _ = self._filters(q)
        joins = "FROM sales x JOIN products p ON p.product_id = x.product_id JOIN stores s ON s.store_id = x.store_id"
        w = f"WHERE {' AND '.join(where)}" if where else ""

        if re.search(r"reorder|low stock|running low|restock", q):
            city_where = [c for c in where if c.startswith("s.city")]
            city_params = [p for p, c in zip(params, where) if c.startswith("s.city")]
            cw = " AND " + " AND ".join(city_where) if city_where else ""
            return (
                "SELECT s.city, p.name, i.on_hand, i.reorder_point FROM inventory i "
                "JOIN products p ON p.product_id = i.product_id JOIN stores s ON s.store_id = i.store_id "
                f"WHERE i.on_hand <= i.reorder_point{cw} "
                "ORDER BY (i.reorder_point - i.on_hand) DESC, s.city, p.name",
                tuple(city_params),
            )
        if re.search(r"\b(top|best[- ]selling|best)\b", q):
            n = int(m.group(1)) if (m := re.search(r"top\s+(\d+)", q)) else 5
            metric = "SUM(x.quantity)" if re.search(r"\bunits?\b|quantity", q) else "ROUND(SUM(x.revenue), 2)"
            alias = "units" if "quantity" in metric else "revenue"
            return (f"SELECT p.name, {metric} AS {alias} {joins} {w} GROUP BY p.name ORDER BY {alias} DESC LIMIT {n}", tuple(params))
        if re.search(r"by store|each store|which store|per store", q):
            return (f"SELECT s.city, ROUND(SUM(x.revenue), 2) AS revenue {joins} {w} GROUP BY s.city ORDER BY revenue DESC", tuple(params))
        if re.search(r"by category|each category|per category", q):
            return (f"SELECT p.category, ROUND(SUM(x.revenue), 2) AS revenue {joins} {w} GROUP BY p.category ORDER BY revenue DESC", tuple(params))
        if re.search(r"by month|monthly|trend|each month", q):
            return (f"SELECT substr(x.sale_date, 1, 7) AS month, ROUND(SUM(x.revenue), 2) AS revenue {joins} {w} GROUP BY month ORDER BY month", tuple(params))
        if re.search(r"total|revenue|sales", q):
            return (f"SELECT ROUND(SUM(x.revenue), 2) AS total_revenue, SUM(x.quantity) AS units {joins} {w}", tuple(params))
        return None

    # ---- LLM path ---------------------------------------------------------
    def _llm_sql(self, question: str) -> DataAnswer:
        prompt, last_error = f"Question: {question}", None
        for attempt in range(2):  # one self-correction retry
            with self.tracer.span("llm_sql", attempt=attempt):
                raw = self.llm.complete(SYSTEM, prompt, max_tokens=400).text
            try:
                sql = validate_sql(raw)
                cols, rows = self.run_sql(sql)
                return DataAnswer(format_table(cols, rows), sql, cols, rows)
            except (UnsafeSQLError, sqlite3.Error) as e:
                last_error = str(e)
                prompt = f"Question: {question}\nYour previous SQL:\n{raw}\nfailed with: {e}\nReturn corrected SQL only."
        return DataAnswer("I couldn't build a safe query for that question.", error=last_error)

    def answer(self, question: str) -> DataAnswer:
        with self.tracer.span("data_agent") as span:
            if self.llm is not None:
                result = self._llm_sql(question)
            else:
                tpl = self.template_sql(question)
                if tpl is None:
                    result = DataAnswer(
                        "Offline mode can't map that question to a query yet. Set LLM_PROVIDER to use text-to-SQL.",
                        error="no_template",
                    )
                else:
                    sql, params = tpl
                    cols, rows = self.run_sql(validate_sql(sql), params)
                    result = DataAnswer(format_table(cols, rows), sql, cols, rows)
            span.attrs["sql"] = result.sql
            span.attrs["error"] = result.error
            return result
