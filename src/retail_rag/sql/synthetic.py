"""Deterministic synthetic operational data for the fictional Harbourline Retail.

The tables are designed to pair with the synthetic policy documents, so that
some questions need both: the replenishment policy defines "at risk" (below the
reorder point on two consecutive daily snapshots), and ``inventory_snapshots``
holds two consecutive snapshots; the store handbook defines the shrink
escalation rule (above 1.8% of sales for three consecutive weeks) and
``shrink_weekly`` holds weekly shrink; the click-and-collect 2-hour picking
SLA is checkable against ``orders.released_at`` / ``orders.staged_at``.

Everything is generated from a fixed seed, so the same questions always have
the same answers and the SQL evaluation's gold queries stay valid.
"""

from __future__ import annotations

import random
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20260901
ORDERS_START = date(2026, 6, 1)
ORDERS_END = date(2026, 8, 31)  # inclusive: June, July, August 2026
SNAPSHOT_DATES = ("2026-09-01", "2026-09-02")

SCHEMA = """
CREATE TABLE stores (
    store_id     TEXT PRIMARY KEY,          -- e.g. 'HB001'
    name         TEXT NOT NULL,
    suburb       TEXT NOT NULL,
    state        TEXT NOT NULL,             -- NSW, VIC, QLD, SA, ACT, TAS
    region_type  TEXT NOT NULL,             -- 'metro' or 'regional'
    format       TEXT NOT NULL,             -- 'full_line', 'metro', 'express'
    opened_year  INTEGER NOT NULL
);
CREATE TABLE products (
    sku              TEXT PRIMARY KEY,      -- e.g. 'SKU-1001'
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    velocity_class   TEXT NOT NULL,         -- 'fast', 'medium', 'slow'
    supply_route     TEXT NOT NULL,         -- 'direct' (to store) or 'dc' (distribution centre)
    shelf_life_days  INTEGER NOT NULL,
    is_chilled       INTEGER NOT NULL,      -- 1 = chilled or frozen
    unit_price_aud   REAL NOT NULL
);
CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    store_id      TEXT NOT NULL REFERENCES stores(store_id),
    order_date    TEXT NOT NULL,            -- ISO date, 2026-06-01 .. 2026-08-31
    channel       TEXT NOT NULL,            -- 'in_store', 'click_and_collect', 'home_delivery'
    loyalty_tier  TEXT NOT NULL,            -- 'none', 'Member', 'Silver', 'Gold'
    released_at   TEXT,                     -- click and collect only: released to store for picking
    staged_at     TEXT                      -- click and collect only: picked and staged
);
CREATE TABLE order_items (
    order_id        TEXT NOT NULL REFERENCES orders(order_id),
    sku             TEXT NOT NULL REFERENCES products(sku),
    quantity        INTEGER NOT NULL,
    unit_price_aud  REAL NOT NULL,
    line_total_aud  REAL NOT NULL
);
CREATE TABLE inventory_snapshots (
    snapshot_date  TEXT NOT NULL,           -- daily snapshots: 2026-09-01 and 2026-09-02
    store_id       TEXT NOT NULL REFERENCES stores(store_id),
    sku            TEXT NOT NULL REFERENCES products(sku),
    on_hand        INTEGER NOT NULL,
    reorder_point  INTEGER NOT NULL,
    safety_stock   INTEGER NOT NULL,
    PRIMARY KEY (snapshot_date, store_id, sku)
);
CREATE TABLE shrink_weekly (
    store_id    TEXT NOT NULL REFERENCES stores(store_id),
    week_start  TEXT NOT NULL,              -- Monday of the week
    sales_aud   REAL NOT NULL,
    shrink_aud  REAL NOT NULL,              -- value of unexplained stock loss
    PRIMARY KEY (store_id, week_start)
);
CREATE INDEX idx_orders_store_date ON orders(store_id, order_date);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_sku ON order_items(sku);
"""

STORES = [
    ("HB001", "Harbourline Parramatta", "Parramatta", "NSW", "metro", "full_line", 2011),
    ("HB002", "Harbourline Chatswood", "Chatswood", "NSW", "metro", "metro", 2014),
    ("HB003", "Harbourline Newcastle", "Newcastle", "NSW", "regional", "full_line", 2009),
    ("HB004", "Harbourline Wollongong", "Wollongong", "NSW", "regional", "full_line", 2016),
    ("HB005", "Harbourline Richmond", "Richmond", "VIC", "metro", "metro", 2012),
    ("HB006", "Harbourline Geelong", "Geelong", "VIC", "regional", "full_line", 2010),
    ("HB007", "Harbourline Docklands", "Docklands", "VIC", "metro", "express", 2021),
    ("HB008", "Harbourline Fortitude Valley", "Fortitude Valley", "QLD", "metro", "metro", 2015),
    ("HB009", "Harbourline Townsville", "Townsville", "QLD", "regional", "full_line", 2013),
    ("HB010", "Harbourline Norwood", "Norwood", "SA", "metro", "full_line", 2018),
    ("HB011", "Harbourline Braddon", "Braddon", "ACT", "metro", "express", 2022),
    ("HB012", "Harbourline Launceston", "Launceston", "TAS", "regional", "full_line", 2019),
]

# name, category, velocity, supply route, shelf life (days), chilled, price
PRODUCTS = [
    ("Full cream milk 2L", "Dairy", "fast", "direct", 10, 1, 3.60),
    ("Greek yoghurt 1kg", "Dairy", "medium", "direct", 21, 1, 6.50),
    ("Cheddar block 500g", "Dairy", "fast", "dc", 90, 1, 8.00),
    ("Free range eggs 12pk", "Dairy", "fast", "direct", 35, 1, 7.20),
    ("Butter 500g", "Dairy", "medium", "dc", 120, 1, 6.80),
    ("Sourdough loaf", "Bakery", "fast", "direct", 4, 0, 5.50),
    ("Wholemeal bread 700g", "Bakery", "fast", "direct", 6, 0, 4.20),
    ("Croissants 4pk", "Bakery", "medium", "direct", 3, 0, 6.00),
    ("Bananas 1kg", "Produce", "fast", "dc", 6, 0, 3.90),
    ("Royal gala apples 1kg", "Produce", "fast", "dc", 21, 0, 5.00),
    ("Baby spinach 120g", "Produce", "medium", "dc", 5, 1, 3.50),
    ("Avocado each", "Produce", "medium", "dc", 5, 0, 2.20),
    ("Frozen peas 1kg", "Frozen", "medium", "dc", 365, 1, 3.80),
    ("Vanilla ice cream 2L", "Frozen", "medium", "dc", 365, 1, 7.50),
    ("Frozen pizza 400g", "Frozen", "slow", "dc", 270, 1, 6.00),
    ("Chicken breast fillets 1kg", "Meat", "fast", "direct", 5, 1, 13.00),
    ("Beef mince 500g", "Meat", "fast", "direct", 4, 1, 8.50),
    ("Atlantic salmon 2pk", "Seafood", "medium", "direct", 4, 1, 12.00),
    ("Pasta 500g", "Pantry", "fast", "dc", 540, 0, 2.00),
    ("Basmati rice 2kg", "Pantry", "medium", "dc", 540, 0, 7.00),
    ("Tinned tomatoes 400g", "Pantry", "fast", "dc", 720, 0, 1.40),
    ("Extra virgin olive oil 1L", "Pantry", "slow", "dc", 540, 0, 14.00),
    ("Peanut butter 375g", "Pantry", "medium", "dc", 365, 0, 4.50),
    ("Breakfast cereal 500g", "Pantry", "medium", "dc", 300, 0, 5.80),
    ("Ground coffee 200g", "Pantry", "medium", "dc", 365, 0, 9.00),
    ("Sparkling water 10pk", "Beverages", "medium", "dc", 365, 0, 7.00),
    ("Orange juice 2L", "Beverages", "medium", "direct", 14, 1, 6.20),
    ("Cola 1.25L", "Beverages", "fast", "dc", 270, 0, 3.20),
    ("Toilet paper 12pk", "Household", "medium", "dc", 1000, 0, 9.50),
    ("Dishwashing liquid 1L", "Household", "slow", "dc", 1000, 0, 4.00),
    ("Laundry powder 2kg", "Household", "slow", "dc", 1000, 0, 16.00),
    ("Razor cartridges 4pk", "Health & Beauty", "slow", "dc", 1000, 0, 22.00),
    ("Vitamin C 60 tablets", "Health & Beauty", "slow", "dc", 720, 0, 12.50),
    ("Baby formula 900g", "Baby", "slow", "dc", 540, 0, 28.00),
    ("Nappies 50pk", "Baby", "medium", "dc", 1000, 0, 24.00),
    ("Dog food 3kg", "Pet", "slow", "dc", 365, 0, 19.00),
]

CHANNEL_WEIGHTS = (("in_store", 0.72), ("click_and_collect", 0.18), ("home_delivery", 0.10))
TIER_WEIGHTS = (("none", 0.45), ("Member", 0.30), ("Silver", 0.17), ("Gold", 0.08))
VELOCITY_WEIGHT = {"fast": 6.0, "medium": 2.5, "slow": 0.8}


def _choice(rng: random.Random, weighted: tuple[tuple[str, float], ...]) -> str:
    return rng.choices([name for name, _ in weighted], [weight for _, weight in weighted])[0]


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


Product = tuple[str, str, str, str, str, int, int, float]
Row = tuple[object, ...]
STORE_SIZE = {row[0]: {"full_line": 1.0, "metro": 0.7, "express": 0.4}[row[5]] for row in STORES}


def _products() -> list[Product]:
    return [(f"SKU-{1001 + index}", *row) for index, row in enumerate(PRODUCTS)]


def _click_and_collect_times(rng: random.Random, day: date, store_id: str) -> tuple[str, str]:
    start = datetime(day.year, day.month, day.day, rng.randint(7, 19), rng.randint(0, 59))
    # Most orders are staged well inside the 2-hour SLA; two regional stores run
    # slower, so "which stores miss the SLA" has a non-trivial answer.
    mean_minutes = 95 if store_id in {"HB003", "HB009"} else 55
    minutes = int(rng.expovariate(1 / mean_minutes)) + 10
    staged = start + timedelta(minutes=minutes)
    return start.isoformat(timespec="minutes"), staged.isoformat(timespec="minutes")


def _orders(rng: random.Random, products: list[Product]) -> tuple[list[Row], list[Row]]:
    weights = [VELOCITY_WEIGHT[product[3]] for product in products]
    orders: list[Row] = []
    items: list[Row] = []
    for day in _days(ORDERS_START, ORDERS_END):
        weekend_boost = 1.25 if day.weekday() >= 5 else 1.0
        for store_id, *_ in STORES:
            count = max(int(rng.gauss(14, 3) * STORE_SIZE[store_id] * weekend_boost), 1)
            for _ in range(count):
                order_id = f"O{len(orders) + 1:06d}"
                channel = _choice(rng, CHANNEL_WEIGHTS)
                released, staged = (
                    _click_and_collect_times(rng, day, store_id)
                    if channel == "click_and_collect"
                    else (None, None)
                )
                tier = _choice(rng, TIER_WEIGHTS)
                orders.append(
                    (order_id, store_id, day.isoformat(), channel, tier, released, staged)
                )
                for product in rng.choices(products, weights, k=rng.randint(1, 6)):
                    quantity = rng.choice((1, 1, 1, 2, 2, 3))
                    price = product[7]
                    items.append(
                        (order_id, product[0], quantity, price, round(price * quantity, 2))
                    )
    return orders, items


def _snapshot_levels(rng: random.Random, reorder_point: int) -> tuple[int, int]:
    roll = rng.random()
    if roll < 0.04:  # at risk: below the reorder point on both snapshots
        return rng.randint(0, reorder_point - 1), rng.randint(0, reorder_point - 1)
    if roll < 0.07:  # dips below for one day only: not at risk under the policy
        return rng.randint(0, reorder_point - 1), reorder_point + rng.randint(1, 20)
    day_one = max(rng.randint(reorder_point - 4, reorder_point * 3), reorder_point)
    return day_one, max(day_one - rng.randint(0, 6), reorder_point)


def _snapshots(rng: random.Random, products: list[Product]) -> list[Row]:
    rows: list[Row] = []
    for store_id, *_ in STORES:
        for product in products:
            velocity = product[3]
            safety = {"fast": 9, "medium": 6, "slow": 4}[velocity]
            reorder_point = safety + {"fast": 24, "medium": 10, "slow": 3}[velocity]
            day_one, day_two = _snapshot_levels(rng, reorder_point)
            if rng.random() < 0.015:
                day_two = 0  # out of stock at the second snapshot
            rows.append((SNAPSHOT_DATES[0], store_id, product[0], day_one, reorder_point, safety))
            rows.append((SNAPSHOT_DATES[1], store_id, product[0], day_two, reorder_point, safety))
    return rows


def _shrink(rng: random.Random) -> list[Row]:
    rows: list[Row] = []
    weeks = [ORDERS_START + timedelta(days=7 * n) for n in range(13)]  # 2026-06-01 is a Monday
    for store_id, *_ in STORES:
        for index, week in enumerate(weeks):
            sales = round(rng.uniform(180_000, 260_000) * STORE_SIZE[store_id], 2)
            rate = rng.uniform(0.008, 0.016)
            if store_id == "HB006" and 8 <= index <= 10:
                rate = rng.uniform(0.019, 0.024)  # three consecutive weeks above 1.8%
            if store_id == "HB002" and index in {4, 9}:
                rate = rng.uniform(0.019, 0.022)  # above 1.8%, but not consecutive
            rows.append((store_id, week.isoformat(), sales, round(sales * rate, 2)))
    return rows


def build_database(path: Path, *, seed: int = SEED) -> Path:
    """(Re)create the SQLite database at ``path`` and return it (atomically replaced)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data, not security
    products = _products()
    orders, items = _orders(rng, products)
    with closing(sqlite3.connect(tmp)) as conn, conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO stores VALUES (?, ?, ?, ?, ?, ?, ?)", STORES)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", products)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
        conn.executemany(
            "INSERT INTO inventory_snapshots VALUES (?, ?, ?, ?, ?, ?)", _snapshots(rng, products)
        )
        conn.executemany("INSERT INTO shrink_weekly VALUES (?, ?, ?, ?)", _shrink(rng))
    tmp.replace(path)
    return path
