from llm.deepseek_sql import generate_sql, fallback_sql
from db.postgres import safe_run_query
from core.sql_validator import validate_sql
from llm.explainer import explain
from core.join_engine import get_join_condition
from core.insight import generate_insight


# 🔥 Extract tables from SQL
def extract_tables(sql: str):
    tables = []

    tokens = sql.replace(",", " ").split()
    for i, token in enumerate(tokens):
        if token.upper() in ["FROM", "JOIN"]:
            if i + 1 < len(tokens):
                tables.append(tokens[i + 1])

    return list(set(tables))


# 🔥 FIXED Detect aggregation (IMPORTANT)
def detect_aggregation(question: str):
    q = question.lower()

    metrics = []
    group_by = []

    # 🔥 detect aggregation intent FIRST
    agg_keywords = ["total", "sum", "average", "avg", "count", "compare"]

    has_agg = any(k in q for k in agg_keywords)

    # 🔥 metrics only if aggregation exists
    if has_agg:
        if "revenue" in q or "sales" in q:
            metrics.append("SUM(fct_transactions.revenue) as total_revenue")

        if "promo" in q:
            metrics.append("SUM(fct_promotions.promo_cost) as total_promo_cost")

    # 🔥 group_by ONLY if aggregation exists
    if has_agg:
        if "by city" in q or "per city" in q:
            group_by.append("dim_users.city")

        if "by product" in q:
            group_by.append("dim_products.product_name")

    return metrics, group_by


# 🔥 Build SELECT clause
def build_select_clause(tables, metrics, group_by):

    # 🔥 If aggregation → ONLY use group_by + metrics
    if metrics and group_by:
        return ",\n".join(group_by + metrics)

    # 🔥 fallback (raw mode)
    select_parts = []

    for t in tables:
        if t == "fct_transactions":
            select_parts += [
                f"{t}.order_id",
                f"{t}.user_id",
                f"{t}.product_id",
                f"{t}.order_date",
                f"{t}.city as transaction_city",
                f"{t}.quantity",
                f"{t}.price",
                f"{t}.revenue",
            ]

        elif t == "dim_users":
            select_parts += [
                f"{t}.city as user_city",
            ]

        elif t == "dim_products":
            select_parts += [
                f"{t}.product_name",
                f"{t}.category",
            ]

        elif t == "fct_promotions":
            select_parts += [
                f"{t}.promo_cost",
            ]

    return ",\n".join(select_parts)


def run_pipeline(question: str):
    sql = generate_sql(question)

    print("🧠 RAW SQL:", sql)

    # 🔥 VALIDATE
    if not validate_sql(sql):
        print("⚠️ Invalid SQL → fallback")
        sql = fallback_sql()

    # 🔥 EXTRACT TABLES
    tables = extract_tables(sql)
    print("📦 TABLES BEFORE:", tables)

    # 🔥 FIX TABLE NAMES
    table_mapping = {
        "daily_transactions": "fct_transactions",
        "transactions": "fct_transactions",
        "users": "dim_users",
        "products": "dim_products",
    }

    tables = [table_mapping.get(t, t) for t in tables]

    # 🔥 SEMANTIC RULES
    q = question.lower()

    if "city" in q or "user" in q:
        if "dim_users" not in tables:
            tables.append("dim_users")

    if "product" in q:
        if "dim_products" not in tables:
            tables.append("dim_products")

    if "promo" in q:
        if "fct_promotions" not in tables:
            tables.append("fct_promotions")

    # Ensure base table
    if "fct_transactions" not in tables:
        tables.insert(0, "fct_transactions")

    print("📦 TABLES AFTER:", tables)

    # 🔥 Detect aggregation
    metrics, group_by = detect_aggregation(question)

    # 🔥 Build SELECT
    select_clause = build_select_clause(tables, metrics, group_by)
    base_table = tables[0]

    query = f"""
    SELECT
    {select_clause}
    FROM {base_table}
    """

    # 🔥 APPLY JOINS (MULTI-HOP)
    joined_tables = {base_table}

    for t in tables[1:]:

        join_condition = get_join_condition(base_table, t)

        if join_condition:
            print(f"🔗 DIRECT JOIN {base_table} → {t}")
            query += f"""
            LEFT JOIN {t}
            ON {join_condition}
            """
            joined_tables.add(t)
            continue

        if t == "fct_promotions":

            if "dim_products" not in joined_tables:
                bridge_join = get_join_condition(base_table, "dim_products")

                if bridge_join:
                    print(f"🔗 BRIDGE JOIN {base_table} → dim_products")
                    query += f"""
                    LEFT JOIN dim_products
                    ON {bridge_join}
                    """
                    joined_tables.add("dim_products")

            promo_join = get_join_condition("dim_products", "fct_promotions")

            if promo_join:
                print("🔗 JOIN dim_products → fct_promotions")
                query += f"""
                LEFT JOIN fct_promotions
                ON {promo_join}
                """
                joined_tables.add("fct_promotions")

    # 🔥 GROUP BY only when valid
    if metrics and group_by:
        query += f"""
        GROUP BY {", ".join(group_by)}
        """

    # 🔥 LIMIT only for raw
    if not metrics:
        query += "\nLIMIT 1000"

    sql = query

    print("🚀 FINAL SQL:", sql)

    # 🔥 EXECUTE
    result = safe_run_query(sql)

    if isinstance(result, tuple) or result is None:
        return {
            "sql": sql,
            "error": result[1] if isinstance(result, tuple) else "Unknown DB error"
        }

    try:
        insight = generate_insight(result, question)
    except Exception as e:
        insight = f"Insight failed: {str(e)}"

    return {
        "sql": sql,
        "data": result,
        "insight": insight
    }