from llm.deepseek_sql import generate_sql, fallback_sql
from db.postgres import safe_run_query
from core.sql_validator import validate_sql
from core.join_engine import get_join_condition
from core.insight import generate_insight


def extract_tables(sql: str):
    tables = []
    tokens = sql.replace(",", " ").split()
    for i, token in enumerate(tokens):
        if token.upper() in ["FROM", "JOIN"]:
            if i + 1 < len(tokens):
                tables.append(tokens[i + 1])
    # dict.fromkeys dedupes while preserving first-seen order -- a plain
    # set() here was the bug: it scrambles order, and everything below
    # assumes tables[0] is the base/hub table (fct_transactions). When set()
    # happened to put dim_products or dim_users first instead, the query
    # got built FROM the wrong table and dim_users had no join path to it,
    # producing "missing FROM-clause entry for table dim_users".
    return list(dict.fromkeys(tables))


def detect_aggregation(question: str):
    q = question.lower()

    metrics = []
    group_by = []

    agg_keywords = ["total", "sum", "average", "avg", "count", "compare"]
    has_agg = any(k in q for k in agg_keywords)

    if has_agg:
        if "revenue" in q or "sales" in q:
            metrics.append("SUM(fct_transactions.revenue) as total_revenue")
        if "promo" in q:
            metrics.append("SUM(fct_promotions.promo_cost) as total_promo_cost")

    if has_agg:
        if "by city" in q or "per city" in q:
            group_by.append("dim_users.city")
        if "by product" in q:
            group_by.append("dim_products.product_name")

    return metrics, group_by


def build_select_clause(tables, metrics, group_by):
    if metrics and group_by:
        return ",\n".join(group_by + metrics)

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
            select_parts += [f"{t}.city as user_city"]
        elif t == "dim_products":
            select_parts += [f"{t}.product_name", f"{t}.category"]
        elif t == "fct_promotions":
            select_parts += [f"{t}.promo_cost"]

    return ",\n".join(select_parts)


def run_pipeline(question: str):
    sql = generate_sql(question)
    print("RAW SQL:", sql)

    if not validate_sql(sql):
        print("Invalid SQL -> fallback")
        sql = fallback_sql()

    tables = extract_tables(sql)
    print("TABLES BEFORE:", tables)

    # NOTE: no more table_mapping dict here. db/postgres.ALLOWED_TABLES,
    # core/join_engine.JOIN_MAP, and the dbt mart models in retail_dbt/
    # all use the same table names now (fct_transactions, dim_users,
    # dim_products, fct_promotions), so nothing needs translating.

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

    if "fct_transactions" not in tables:
        tables.insert(0, "fct_transactions")

    print("TABLES AFTER:", tables)

    metrics, group_by = detect_aggregation(question)
    select_clause = build_select_clause(tables, metrics, group_by)
    base_table = tables[0]

    query = f"""
    SELECT
    {select_clause}
    FROM {base_table}
    """

    joined_tables = {base_table}

    for t in tables[1:]:
        join_condition = get_join_condition(base_table, t)

        if join_condition:
            print(f"DIRECT JOIN {base_table} -> {t}")
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
                    print(f"BRIDGE JOIN {base_table} -> dim_products")
                    query += f"""
                    LEFT JOIN dim_products
                    ON {bridge_join}
                    """
                    joined_tables.add("dim_products")

            promo_join = get_join_condition("dim_products", "fct_promotions")
            if promo_join:
                print("JOIN dim_products -> fct_promotions")
                query += f"""
                LEFT JOIN fct_promotions
                ON {promo_join}
                """
                joined_tables.add("fct_promotions")

    if metrics and group_by:
        query += f"""
        GROUP BY {", ".join(group_by)}
        """

    if not metrics:
        query += "\nLIMIT 1000"

    sql = query
    print("FINAL SQL:", sql)

    result = safe_run_query(sql)

    if isinstance(result, tuple) or result is None:
        return {
            "sql": sql,
            "error": result[1] if isinstance(result, tuple) else "Unknown DB error",
        }

    try:
        insight = generate_insight(result, question)
    except Exception as e:
        insight = f"Insight failed: {str(e)}"

    return {"sql": sql, "data": result, "insight": insight}