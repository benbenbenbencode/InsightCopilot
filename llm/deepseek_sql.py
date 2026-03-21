import subprocess
import re
from db.postgres import get_schema


def build_schema_text():
    schema = get_schema()

    text = ""
    for table, cols in schema.items():
        text += f"\nTable: {table}\n"
        for c in cols:
            text += f"- {c}\n"

    return text


def extract_sql(text: str) -> str:
    # ✅ case 1: ```sql ... ```
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # ✅ case 2: SELECT ... ;
    match = re.search(r"(SELECT[\s\S]+?;)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def fallback_sql():
    return """SELECT product, SUM(revenue) AS total_revenue
FROM daily_transactions
GROUP BY product
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
- NO comments
- LIMIT 100

Schema:
{schema_text}

Example:
SELECT product, SUM(revenue) AS total_revenue
FROM daily_transactions
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 100;

Question:
{question}
"""

    result = subprocess.run(
        ["ollama", "run", "deepseek-coder"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    raw_output = result.stdout.strip()

    # 🔥 EXTRACT CLEAN SQL
    sql = extract_sql(raw_output)

    # 🔥 FIX BAD LIMIT
    sql = re.sub(r"LIMIT\s*\{.*?\}", "LIMIT 100", sql)

    # 🔥 VALIDATE MINIMUM STRUCTURE
    if not all(k in sql.upper() for k in ["SELECT", "FROM"]):
        return fallback_sql()

    if "SUM(" in sql.upper() and "GROUP BY" not in sql.upper():
        return fallback_sql()

    if "JOIN" in sql.upper():
        return fallback_sql()

    return sql.strip()