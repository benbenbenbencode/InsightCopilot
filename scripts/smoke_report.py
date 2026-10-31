"""End-to-end report smoke test: question -> files on disk."""

import sys
import time

from report.builder import answer_question, export

QUESTION = "What is total revenue by product category, and which one is underperforming?"


def main() -> int:
    started = time.time()
    answer = answer_question(QUESTION)
    print(f"answer ok={answer.ok} in {time.time() - started:.1f}s")
    print(f"timings: {answer.timings}")

    if not answer.ok:
        print("ERROR:", answer.error)
        print("SQL:", answer.sql)
        return 1

    bundle = answer.bundle
    print("\nSQL:", bundle.sql[:200])
    print("rows:", len(bundle.df))
    print("\nKPIs:")
    for kpi in bundle.analysis.kpis:
        print(f"  {kpi.label}: {kpi.display}")
    print(f"\nfindings: {len(bundle.analysis.findings)}")
    for finding in bundle.analysis.findings:
        print(f"  [{finding.severity}] {finding.title}")
    print(f"\nheadline: {bundle.narrative.headline}")
    print(f"summary: {bundle.narrative.summary}")
    print(f"key points: {len(bundle.narrative.key_points)}")
    for point in bundle.narrative.key_points:
        print(f"  - {point}")
    print(f"recommendations: {len(bundle.narrative.recommendations)}")
    for rec in bundle.narrative.recommendations:
        print(f"  * {rec}")
    print(f"grounded: {bundle.narrative.grounded} {bundle.narrative.ungrounded_numbers}")

    files = export(bundle, formats=("pdf", "pptx", "xlsx"))
    print("\nfiles:")
    for kind, path in files.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
