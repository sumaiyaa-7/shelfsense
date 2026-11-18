"""Build a synthetic, reproducible retail database (SQLite).

Five stores, twenty products, daily sales for Q1 2026, and current inventory
with reorder points. Seeded so every run (and every eval) sees identical data.

    python -m shelfsense.seed_db
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .config import get_settings

STORES = [
    (1, "Edison Route 1", "Edison", "NJ"),
    (2, "Boston Downtown", "Boston", "MA"),
    (3, "Charlotte SouthPark", "Charlotte", "NC"),
    (4, "Cincinnati Kenwood", "Cincinnati", "OH"),
    (5, "Santa Clara Central", "Santa Clara", "CA"),
]

# (name, category, unit_price, unit_cost, base daily demand)
PRODUCTS = [
    ("Copy Paper 10-Ream Case", "Paper", 54.99, 38.00, 6.0),
    ("Premium Notebook 3-Pack", "Paper", 12.99, 5.10, 9.0),
    ("Sticky Notes 12-Pack", "Paper", 9.49, 3.20, 11.0),
    ("Black Ink Cartridge XL", "Ink & Toner", 42.99, 21.00, 5.0),
    ("Color Ink Cartridge Combo", "Ink & Toner", 64.99, 33.00, 3.5),
    ("Laser Toner Cartridge", "Ink & Toner", 89.99, 47.00, 2.0),
    ("Gel Pens 12-Pack", "Writing", 14.99, 5.50, 8.0),
    ("Highlighters 6-Pack", "Writing", 7.99, 2.60, 7.0),
    ("Dry Erase Markers 8-Pack", "Writing", 11.49, 4.10, 5.0),
    ("Wireless Mouse", "Technology", 24.99, 11.00, 4.0),
    ("Mechanical Keyboard", "Technology", 79.99, 41.00, 1.5),
    ("27-inch Monitor", "Technology", 229.99, 160.00, 0.8),
    ("All-in-One Inkjet Printer", "Technology", 149.99, 98.00, 0.9),
    ("USB-C Hub", "Technology", 39.99, 17.50, 2.5),
    ("Ergonomic Office Chair", "Furniture", 249.99, 140.00, 0.6),
    ("Standing Desk Converter", "Furniture", 189.99, 110.00, 0.4),
    ("Desk Organizer", "Furniture", 19.99, 7.80, 2.0),
    ("Shipping Boxes 25-Pack", "Shipping", 29.99, 12.00, 3.0),
    ("Packing Tape 6-Roll", "Shipping", 16.99, 6.40, 4.5),
    ("Bubble Mailers 50-Pack", "Shipping", 24.49, 9.90, 2.8),
]

STORE_FACTOR = {1: 1.15, 2: 1.30, 3: 0.95, 4: 0.85, 5: 1.05}
SCHEMA = """
CREATE TABLE stores (store_id INTEGER PRIMARY KEY, name TEXT, city TEXT, state TEXT);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price REAL, unit_cost REAL);
CREATE TABLE inventory (store_id INTEGER, product_id INTEGER, on_hand INTEGER, reorder_point INTEGER,
                        PRIMARY KEY (store_id, product_id));
CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, sale_date TEXT, store_id INTEGER, product_id INTEGER,
                    quantity INTEGER, revenue REAL);
"""


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm; fine for small lambdas and keeps us stdlib-only.
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def build(db_path: Path, seed: int = 42) -> Path:
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO stores VALUES (?,?,?,?)", STORES)
    con.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?)",
        [(i + 1, n, c, p, cost) for i, (n, c, p, cost, _) in enumerate(PRODUCTS)],
    )

    rows, sale_id, d = [], 1, date(2026, 1, 1)
    while d <= date(2026, 3, 31):
        weekday_boost = 1.25 if d.weekday() >= 5 else 1.0
        month_boost = {1: 1.20, 2: 0.90, 3: 1.00}[d.month]  # back-to-school-style January bump
        for store_id, *_ in STORES:
            for pid, (_, _, price, _, demand) in enumerate(PRODUCTS, start=1):
                qty = _poisson(rng, demand * STORE_FACTOR[store_id] * weekday_boost * month_boost)
                if qty:
                    rows.append((sale_id, d.isoformat(), store_id, pid, qty, round(qty * price, 2)))
                    sale_id += 1
        d += timedelta(days=1)
    con.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", rows)

    inv = []
    for store_id, *_ in STORES:
        for pid, (*_, demand) in enumerate(PRODUCTS, start=1):
            reorder_point = max(3, round(demand * 7))
            # ~20% of SKUs are running low, so the "low stock" question has real answers
            on_hand = rng.randint(0, reorder_point) if rng.random() < 0.2 else rng.randint(reorder_point + 1, reorder_point * 4)
            inv.append((store_id, pid, on_hand, reorder_point))
    con.executemany("INSERT INTO inventory VALUES (?,?,?,?)", inv)
    con.commit()
    con.close()
    return db_path


if __name__ == "__main__":
    path = build(get_settings().db_path)
    print(f"Built {path}")
