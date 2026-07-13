import re

from db.postgres import ALLOWED_TABLES
from core.join_engine import is_allowed_join

FORBIDDEN_KEYWORDS = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE", "--", ";"]

MAX_LIMIT = 1000


def _extract_tables(sql: str):
    """Pulls every table name that follows FROM or JOIN."""
    tokens = re.sub(r"[,\n]", " ", sql).split()
    tables = []
    for i, token in enumerate(tokens):
        if token.upper() in ("FROM", "JOIN") and i + 1 < len(tokens):
            tables.append(tokens[i + 1].strip().rstrip(";"))
    return tables


def validate_sql(sql: str) -> bool:
    ok, _ = validate_sql_verbose(sql)
    return ok


def validate_sql_verbose(sql: str):
    """Returns (is_valid, reason). Whitelist-based instead of banning every
    JOIN outright: a JOIN is fine as long as every table involved is in
    ALLOWED_TABLES and the join pair is one we recognize in JOIN_MAP.
    """
    sql_upper = sql.upper()

    for word in FORBIDDEN_KEYWORDS:
        if word in sql_upper:
            return False, f"forbidden keyword: {word}"

    if not sql_upper.strip().startswith("SELECT"):
        return False, "must start with SELECT"

    if "FROM" not in sql_upper:
        return False, "missing FROM"

    tables = _extract_tables(sql)
    if not tables:
        return False, "no tables found"

    unknown = [t for t in tables if t not in ALLOWED_TABLES]
    if unknown:
        return False, f"unknown table(s): {unknown}"

    if "JOIN" in sql_upper:
        base = tables[0]
        for t in tables[1:]:
            if not is_allowed_join(base, t):
                return False, f"unrecognized join pair: {base} <-> {t}"

    if "SUM(" in sql_upper and "GROUP BY" not in sql_upper:
        return False, "aggregation without GROUP BY"

    return True, "ok"
