def validate_sql(sql: str) -> bool:
    sql_upper = sql.upper()

    if any(x in sql_upper for x in ["DELETE", "UPDATE", "INSERT", "DROP"]):
        return False

    if "JOIN" in sql_upper:
        return False

    if not sql_upper.startswith("SELECT"):
        return False

    if "FROM" not in sql_upper:
        return False

    if "LIMIT" not in sql_upper:
        return False

    return True