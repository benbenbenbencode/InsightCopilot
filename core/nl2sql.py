"""Natural-language to SQL, grounded in the semantic layer.

Flow:
  1. build a prompt from semantic_layer.yml (tables, joins, metrics, examples)
  2. ask the local model for one SELECT statement
  3. clean it (fences, trailing semicolon, comments)
  4. validate it statically (whitelist of tables, read-only, single statement)
  5. validate it against PostgreSQL with EXPLAIN -- the authoritative check
  6. on failure, feed the exact error back and retry (bounded)

Step 5 is what the old keyword-based builder could not do: it proves the
query is valid against the real schema before anything executes.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from core.executor import explain_query
from llm.client import LLMError, OllamaClient, get_client
from semantic import SemanticLayer
from semantic import load as load_semantic_layer

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|vacuum|call|do)\b",
    re.IGNORECASE,
)

SYSTEM_TEMPLATE = """You are a senior PostgreSQL analytics engineer.
Translate the user's business question into ONE read-only SELECT statement.

Rules:
- Output SQL only. No prose, no markdown fences, no comments.
- Use only the tables, columns and join paths listed in the semantic layer.
- Prefer the metric expressions exactly as defined -- they encode the business definition.
- Use LEFT JOIN with the documented join keys. Never invent a join.
- Always alias aggregates with a readable snake_case name.
- Filter with explicit dates or relative intervals (e.g. CURRENT_DATE - INTERVAL '90 days').
- If the question cannot be answered with this schema, output exactly: CANNOT_ANSWER
- For comparisons, rankings and "which is best/worst" questions, return ALL groups
  ordered by the metric. Never add LIMIT 1 -- the report highlights the winner itself.

{semantic_context}
"""

USER_TEMPLATE = """Question: {question}
SQL:"""


@dataclass
class Attempt:
    sql: str
    ok: bool
    error: str = ""
    stage: str = ""  # "static" | "explain" | "model"

    def to_dict(self) -> dict:
        return {"sql": self.sql, "ok": self.ok, "error": self.error, "stage": self.stage}


@dataclass
class NL2SQLResult:
    sql: str
    question: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    seconds: float = 0.0
    ok: bool = True
    error: str = ""

    @property
    def repaired(self) -> bool:
        return len(self.attempts) > 1 and self.ok

    def to_dict(self) -> dict:
        return {
            "sql": self.sql,
            "question": self.question,
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "attempts": [a.to_dict() for a in self.attempts],
            "repaired": self.repaired,
            "seconds": round(self.seconds, 2),
        }


def clean_sql(text: str) -> str:
    """Strip fences, comments and trailing semicolons from model output."""
    sql = (text or "").strip()

    fence = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1)

    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.replace(" ;", ";")


def static_validate(sql: str, layer: SemanticLayer) -> tuple[bool, str]:
    """Cheap, model-independent checks before touching the database."""
    if not sql:
        return False, "empty SQL"

    if sql.upper().startswith("CANNOT_ANSWER"):
        return False, "the model says this question is not answerable with this schema"

    lowered = sql.lower().strip()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "only SELECT / WITH statements are allowed"

    if FORBIDDEN.search(sql):
        return False, "statement contains a write or DDL keyword"

    # A single statement only: an embedded semicolon means stacked queries.
    if ";" in sql.rstrip(";"):
        return False, "multiple statements are not allowed"

    tables = extract_tables(sql)
    if not tables:
        return False, "no table found in FROM/JOIN"

    allowed = set(layer.table_names())
    unknown = [t for t in tables if t not in allowed]
    if unknown:
        return False, (
            f"unknown table(s): {', '.join(unknown)}. "
            f"Allowed tables: {', '.join(sorted(allowed))}"
        )

    return True, "ok"


TABLE_TOKEN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", re.IGNORECASE)


def extract_tables(sql: str) -> list[str]:
    """Table names following FROM/JOIN, deduped, order preserved."""
    found = TABLE_TOKEN.findall(sql)
    seen: list[str] = []
    for name in found:
        if name.lower() in ("select", "lateral", "unnest"):
            continue
        if name not in seen:
            seen.append(name)
    return seen


def allowed_join_pairs(layer: SemanticLayer) -> set[frozenset]:
    return {frozenset((j.left, j.right)) for j in layer.joins}


def build_system_prompt(layer: SemanticLayer) -> str:
    """The exact system prompt used for inference AND for dataset generation."""
    context = layer.prompt_context()

    # Tell the model what the data actually covers. Without this it anchors
    # relative dates to CURRENT_DATE and silently returns zero rows.
    try:
        from core.profile import coverage_hint

        hint = coverage_hint()
        if hint:
            context = f"{context}\n\n{hint}"
    except Exception:
        pass

    return SYSTEM_TEMPLATE.format(semantic_context=context)


def question_to_sql(
    question: str,
    layer: SemanticLayer | None = None,
    client: OllamaClient | None = None,
    model: str | None = None,
    max_attempts: int = 3,
) -> NL2SQLResult:
    layer = layer or load_semantic_layer()
    client = client or get_client()
    model = model or os.environ.get("SQL_MODEL") or "qwen3.6:latest"

    system = build_system_prompt(layer)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]

    started = time.time()
    attempts: list[Attempt] = []
    sql = ""

    for attempt_no in range(1, max_attempts + 1):
        try:
            result = client.chat(
                messages,
                model=model,
                temperature=0.0 if attempt_no == 1 else 0.2,
                num_ctx=int(os.environ.get("SQL_NUM_CTX", 8192)),
                num_predict=600,
            )
        except LLMError as exc:
            attempts.append(Attempt(sql="", ok=False, error=str(exc), stage="model"))
            return NL2SQLResult(
                sql="",
                question=question,
                model=model,
                attempts=attempts,
                seconds=time.time() - started,
                ok=False,
                error=str(exc),
            )

        sql = clean_sql(result.text)

        ok, reason = static_validate(sql, layer)
        if not ok:
            attempts.append(Attempt(sql=sql, ok=False, error=reason, stage="static"))
            messages.extend(
                [
                    {"role": "assistant", "content": result.text},
                    {
                        "role": "user",
                        "content": f"That SQL is invalid: {reason}\nReturn a corrected SELECT statement only.",
                    },
                ]
            )
            continue

        ok, reason = explain_query(sql)
        if not ok:
            attempts.append(Attempt(sql=sql, ok=False, error=reason, stage="explain"))
            messages.extend(
                [
                    {"role": "assistant", "content": result.text},
                    {
                        "role": "user",
                        "content": (
                            "PostgreSQL rejected that query with:\n"
                            f"{reason}\n"
                            "Fix the columns/joins/types and return only the corrected SELECT."
                        ),
                    },
                ]
            )
            continue

        attempts.append(Attempt(sql=sql, ok=True))
        return NL2SQLResult(
            sql=sql,
            question=question,
            model=model,
            attempts=attempts,
            seconds=time.time() - started,
            ok=True,
        )

    last_error = attempts[-1].error if attempts else "no SQL produced"
    return NL2SQLResult(
        sql=sql,
        question=question,
        model=model,
        attempts=attempts,
        seconds=time.time() - started,
        ok=False,
        error=last_error,
    )
