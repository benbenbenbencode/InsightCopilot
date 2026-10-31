"""Read-only query execution with hard guardrails.

Every query the assistant produces runs through here:

  * a read-only transaction -- the database itself refuses writes
  * a statement timeout -- no runaway query can hang the app
  * a wrapping LIMIT -- the UI can never be handed a million rows
  * EXPLAIN validation -- schema errors surface before execution

This is defence in depth: even if the SQL validator were bypassed, PostgreSQL
would still refuse the write.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd
import psycopg2

from db.postgres import DB_CONFIG

STATEMENT_TIMEOUT_MS = int(os.environ.get("DB_STATEMENT_TIMEOUT_MS", 30000))
MAX_ROWS = int(os.environ.get("DB_MAX_ROWS", 5000))


class QueryError(RuntimeError):
    """A query failed in a way worth showing the user."""


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql: str
    rows: int
    seconds: float
    truncated: bool = False
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "seconds": round(self.seconds, 3),
            "truncated": self.truncated,
            "columns": self.columns,
            "data": self.df.to_dict(orient="records"),
        }


def _connect():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True, autocommit=False)
    return conn


def explain_query(sql: str) -> tuple[bool, str]:
    """Ask PostgreSQL to plan the query without running it.

    Returns (ok, error). This catches bad columns, ambiguous references, type
    errors and missing tables -- things a regex validator cannot know.
    """
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 8000")
            cur.execute(f"EXPLAIN {sql}")
            cur.fetchall()
        conn.rollback()
        return True, "ok"
    except psycopg2.Error as exc:
        message = (exc.pgerror or str(exc)).strip().splitlines()
        detail = message[0] if message else str(exc)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return False, detail
    except Exception as exc:  # connection failures etc.
        return False, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def run_select(sql: str, limit: int | None = None) -> QueryResult:
    """Execute a validated SELECT in a read-only transaction and return a DataFrame."""
    limit = int(limit or MAX_ROWS)
    wrapped = f"SELECT * FROM (\n{sql}\n) AS _ic_result LIMIT {limit + 1}"

    started = time.time()
    conn = None

    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(wrapped)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
        conn.rollback()
    except psycopg2.errors.QueryCanceled as exc:
        raise QueryError(
            f"Query exceeded the {STATEMENT_TIMEOUT_MS / 1000:.0f}s timeout. "
            "Try a narrower date range or fewer dimensions."
        ) from exc
    except psycopg2.Error as exc:
        message = (exc.pgerror or str(exc)).strip().splitlines()
        raise QueryError(message[0] if message else str(exc)) from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    elapsed = time.time() - started
    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]

    df = pd.DataFrame(rows, columns=columns)
    df = _coerce(df)

    return QueryResult(
        df=df,
        sql=sql,
        rows=len(df),
        seconds=elapsed,
        truncated=truncated,
        columns=list(df.columns),
    )


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Make numeric/date columns behave predictably for analysis and charts."""
    if df.empty:
        return df

    for column in df.columns:
        series = df[column]
        if series.dtype == object:
            non_null = series.dropna()
            if non_null.empty:
                continue

            # Numeric-looking decimals come back as Decimal from psycopg2.
            try:
                converted = pd.to_numeric(series, errors="raise")
                df[column] = converted
                continue
            except (ValueError, TypeError):
                pass

            # Date-like strings -> datetime, but only when clearly dates.
            sample = non_null.iloc[0]
            if hasattr(sample, "year") and hasattr(sample, "month"):
                try:
                    df[column] = pd.to_datetime(series)
                except (ValueError, TypeError):
                    pass

    return df


def health() -> dict:
    """Connection + version check for the API/UI status panel."""
    conn = None
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version()")
            database, version = cur.fetchone()
        conn.rollback()
        return {
            "ok": True,
            "database": database,
            "version": version.split(",")[0],
            "host": DB_CONFIG["host"],
            "port": DB_CONFIG["port"],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
