import psycopg2
import pandas as pd
import streamlit as st



DB_CONFIG = {
    "host": "localhost",
    "dbname": "retail",
    "user": "postgres",
    "password": "admin",  # 🔴 CHANGE
    "port": 5432
}


ALLOWED_TABLES = [
    "daily_transactions",
    "user_summary",
    "product_metrics"
]


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def run_query(sql: str):
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

@st.cache_data(ttl=60)
def safe_run_query(sql: str):
    try:
        return run_query(sql)
    except Exception as e:
        print("\n=== DB ERROR ===")
        print(sql)
        print(e)
        print("================\n")
        return None, str(e)


# 🔥 FILTERED SCHEMA
def get_schema():
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

        if table not in schema:
            schema[table] = []

        schema[table].append(row["column_name"])

    return schema