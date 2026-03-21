JOIN_MAP = {
    ("fct_transactions", "dim_users"): ("user_id", "user_id"),
    ("fct_transactions", "dim_products"): ("product_id", "product_id"),
    ("fct_promotions", "dim_products"): ("product_id", "product_id"),
}


def get_join_condition(left_table, right_table):
    key = (left_table, right_table)

    if key in JOIN_MAP:
        left_key, right_key = JOIN_MAP[key]
        return f"{left_table}.{left_key} = {right_table}.{right_key}"

    reverse_key = (right_table, left_table)
    if reverse_key in JOIN_MAP:
        right_key, left_key = JOIN_MAP[reverse_key]
        return f"{left_table}.{left_key} = {right_table}.{right_key}"

    return None