"""Quick end-to-end smoke test for the NL->SQL pipeline.

Runs a handful of business questions through the local model, validates each
query against PostgreSQL, and prints the result rows.
"""

import sys
import time

from core.executor import explain_query, run_select
from core.nl2sql import question_to_sql
from semantic import load

QUESTIONS = [
    "What is total revenue by product category?",
    "Which cities generate the most revenue?",
    "How many distinct users bought something in each city?",
    "What is the average order value per product?",
    "Show monthly revenue for 2025",
]


def main() -> int:
    layer = load()
    print(f"semantic layer: {layer.name}, {len(layer.metrics)} metrics, {len(layer.dimensions)} dims")
    print(f"prompt context: {len(layer.prompt_context())} chars\n")

    failures = 0
    for question in QUESTIONS:
        print("=" * 78)
        print("Q:", question)

        t0 = time.time()
        result = question_to_sql(question, layer=layer)
        elapsed = time.time() - t0

        if not result.ok:
            failures += 1
            print(f"  FAILED after {len(result.attempts)} attempt(s) in {elapsed:.1f}s")
            for attempt in result.attempts:
                print(f"   [{attempt.stage}] {attempt.error}")
            continue

        print(f"  ok in {elapsed:.1f}s, attempts={len(result.attempts)}")
        print("  SQL:", result.sql[:220])

        try:
            query = run_select(result.sql, limit=20)
            print(f"  rows={query.rows} in {query.seconds * 1000:.0f}ms, cols={query.columns}")
            print(query.df.head(5).to_string(index=False))
        except Exception as exc:
            failures += 1
            print(f"  EXECUTION FAILED: {exc}")

    print("\n" + "=" * 78)
    print(f"failures: {failures}/{len(QUESTIONS)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
