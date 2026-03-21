def validate_sql(sql: str) -> bool:
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]

    if not sql.upper().startswith("SELECT"):
        return False

    if any(word in sql.upper() for word in forbidden):
        return False

    return True


def retry_generate_sql(generate_fn, question: str, max_retry: int = 3):
    for attempt in range(max_retry):
        sql = generate_fn(question)

        if validate_sql(sql):
            return sql

        print(f"[RETRY] invalid SQL, attempt {attempt+1}")

    raise Exception("Failed to generate valid SQL")