import os
import time
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config comes from environment now, never hardcoded.
# Copy .env.example to .env and fill in your real values.
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "dbname": os.environ.get("DB_NAME", "retail"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD"),
    "port": int(os.environ.get("DB_PORT", 5432)),
}

if not DB_CONFIG["password"]:
    raise RuntimeError(
        "DB_PASSWORD is not set. Copy .env.example to .env and fill it in "
        "before running the app."
    )

# ---------------------------------------------------------------------------
# Single source of truth for the schema. This now matches the dbt mart layer
# in retail_dbt/models/ (fct_transactions, dim_users, dim_products,
# fct_promotions) so the LLM, the join engine, and the actual database all
# agree on table names. No more translation dictionary in pipeline.py.
# ---------------------------------------------------------------------------
ALLOWED_TABLES = [
    "fct_transactions",
    "dim_users",
    "dim_products",
    "fct_promotions",
]


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def run_query(sql: str):
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Framework-agnostic TTL cache. Replaces the old @st.cache_data dependency,
# which coupled the data layer to Streamlit and broke when this module was
# imported from the FastAPI app.
# ---------------------------------------------------------------------------
_query_cache = {}
_CACHE_TTL_SECONDS = 60


def _cache_get(key):
    entry = _query_cache.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        del _query_cache[key]
        return None
    return value


def _cache_set(key, value):
    _query_cache[key] = (value, time.time() + _CACHE_TTL_SECONDS)


def safe_run_query(sql: str):
    cached = _cache_get(sql)
    if cached is not None:
        return cached

    try:
        result = run_query(sql)
        _cache_set(sql, result)
        return result
    except Exception as e:
        print("\n=== DB ERROR ===")
        print(sql)
        print(e)
        print("================\n")
        return None, str(e)


def get_schema():
    """Returns {table_name: [columns]} for every table in ALLOWED_TABLES.

    This is what gets serialized into the prompt for NL->SQL generation, so
    it MUST stay in sync with ALLOWED_TABLES and with join_engine.JOIN_MAP.
    """
    conn = get_connection()

    query = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
    """

    df = pd.read_sql(query, conn)
    conn.close()

    schema = {}

    for _, row in df.iterrows():
        table = row["table_name"]

        if table not in ALLOWED_TABLES:
            continue

        schema.setdefault(table, []).append(row["column_name"])

    return schema
