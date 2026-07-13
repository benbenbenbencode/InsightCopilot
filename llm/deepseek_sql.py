import os
import subprocess
import re

from db.postgres import get_schema
from core.sql_validator import validate_sql_verbose

SQL_MODEL = os.environ.get("SQL_MODEL", "deepseek-coder")


def build_schema_text():
    schema = get_schema()

    text = ""
    for table, cols in schema.items():
        text += f"\nTable: {table}\n"
        for c in cols:
            text += f"- {c}\n"

    return text


def extract_sql(text: str) -> str:
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"(SELECT[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def fallback_sql():
    return """SELECT dim_products.product_name, SUM(fct_transactions.revenue) AS total_revenue
FROM fct_transactions
LEFT JOIN dim_products
ON fct_transactions.product_id = dim_products.product_id
GROUP BY dim_products.product_name
ORDER BY total_revenue DESC
LIMIT 100;"""


def generate_sql(question: str) -> str:
    schema_text = build_schema_text()

    prompt = f"""
Generate ONLY PostgreSQL SQL.

STRICT:
- ONLY output SQL (no explanation)
- ONLY SELECT
- Use ONLY tables listed
- Use ONLY columns listed
- Joins are allowed ONLY between tables that share a documented key below
- NO comments
- LIMIT 100

Schema:
{schema_text}

Known join keys:
- fct_transactions.user_id = dim_users.user_id
- fct_transactions.product_id = dim_products.product_id
- fct_promotions.product_id = dim_products.product_id

Example:
SELECT dim_products.product_name, SUM(fct_transactions.revenue) AS total_revenue
FROM fct_transactions
LEFT JOIN dim_products ON fct_transactions.product_id = dim_products.product_id
GROUP BY dim_products.product_name
ORDER BY total_revenue DESC
LIMIT 100;

Question:
{question}
"""

    result = subprocess.run(
        ["ollama", "run", SQL_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore",
    )

    raw_output = result.stdout.strip()
    sql = extract_sql(raw_output)
    sql = re.sub(r"LIMIT\s*\{.*?\}", "LIMIT 100", sql)

    # Whitelist-based check. A JOIN no longer causes an automatic reject --
    # only an unknown table, an unrecognized join pair, or a genuinely
    # unsafe statement does.
    is_valid, reason = validate_sql_verbose(sql)
    if not is_valid:
        print(f"[SQL REJECTED] {reason} -- falling back. Raw SQL was:\n{sql}")
        return fallback_sql()

    return sql.strip()
