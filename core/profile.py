"""Data coverage profiling.

The seed dataset covers a single 30-day window in 2026, not "up to today".
Without telling the model that, it writes `CURRENT_DATE - INTERVAL '30 days'`
and silently returns nothing. This module detects the real range and injects it
into the prompt, so relative questions anchor to the data instead of the clock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import psycopg2

from db.postgres import DB_CONFIG

CACHE_TTL = 300
_cache: dict = {"at": 0.0, "value": None}


@dataclass
class Coverage:
    table: str
    column: str
    start: date
    end: date
    rows: int

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def hint(self) -> str:
        return (
            "### Data coverage (important)\n"
            f"{self.table}.{self.column} contains data from {self.start} to {self.end} "
            f"({self.days} days, {self.rows:,} rows).\n"
            "The data is historical and ends before today. For questions like "
            "\"recent\", \"last 30 days\" or \"this month\", anchor to this range with "
            "explicit dates instead of CURRENT_DATE, otherwise the query returns no rows."
        )


def get_coverage(force: bool = False) -> Coverage | None:
    now = time.time()
    if not force and _cache["value"] is not None and now - _cache["at"] < CACHE_TTL:
        return _cache["value"]

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(order_date), MAX(order_date), COUNT(*) FROM fct_transactions")
            start, end, rows = cur.fetchone()
        conn.rollback()
    except Exception:
        return _cache["value"]
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if not start or not end:
        return None

    coverage = Coverage("fct_transactions", "order_date", start, end, int(rows))
    _cache.update({"at": now, "value": coverage})
    return coverage


def coverage_hint() -> str:
    coverage = get_coverage()
    return coverage.hint() if coverage else ""
