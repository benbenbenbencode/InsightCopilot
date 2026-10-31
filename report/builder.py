"""Report orchestration: question -> SQL -> query -> analysis -> narrative -> files.

`answer_question` is the single entry point the API and the UI both call, so
there is exactly one definition of "what an answer is".
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.analysis import Analysis, analyze
from core.executor import QueryError, run_select
from core.narrative import Narrative, generate_narrative
from core.nl2sql import NL2SQLResult, question_to_sql
from report.charts import pick_chart, render_png
from report.model import ReportBundle
from report.pdf import build_pdf
from report.pptx import build_pptx
from report.xlsx import build_xlsx
from semantic import load as load_semantic_layer

REPORT_DIR = Path(os.environ.get("REPORT_DIR") or (Path(__file__).resolve().parent.parent / "reports"))

BUILDERS = {"pdf": build_pdf, "pptx": build_pptx, "xlsx": build_xlsx}


@dataclass
class Answer:
    ok: bool
    question: str
    sql: str = ""
    error: str = ""
    bundle: ReportBundle | None = None
    nl2sql: NL2SQLResult | None = None
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)

    def to_dict(self, max_rows: int = 200) -> dict:
        if not self.ok or self.bundle is None:
            return {
                "ok": False,
                "question": self.question,
                "error": self.error,
                "sql": self.sql,
                "attempts": [a.to_dict() for a in (self.nl2sql.attempts if self.nl2sql else [])],
                "seconds": round(self.seconds, 2),
            }

        bundle = self.bundle
        head = bundle.df.head(max_rows)
        return {
            "ok": True,
            "question": self.question,
            "sql": bundle.sql,
            "analysis": bundle.analysis.to_dict(),
            "narrative": bundle.narrative.to_dict(),
            "chart": bundle.chart_spec,
            "rows": int(len(bundle.df)),
            "data": head.to_dict(orient="records"),
            "repaired": bool(self.nl2sql.repaired) if self.nl2sql else False,
            "model": bundle.narrative.model,
            "seconds": round(self.seconds, 2),
            "timings": self.timings,
        }


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "report").lower()).strip("-")
    return (slug[:limit] or "report").rstrip("-")


def answer_question(
    question: str,
    model: str | None = None,
    with_chart: bool = True,
    with_narrative: bool = True,
) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer(ok=False, question=question, error="Question is empty.")

    started = time.time()
    timings: dict = {}

    layer = load_semantic_layer()

    t0 = time.time()
    nl2sql_result = question_to_sql(question, layer=layer, model=model)
    timings["nl2sql"] = round(time.time() - t0, 2)

    if not nl2sql_result.ok:
        return Answer(
            ok=False,
            question=question,
            sql=nl2sql_result.sql,
            error=nl2sql_result.error,
            nl2sql=nl2sql_result,
            seconds=time.time() - started,
            timings=timings,
        )

    t0 = time.time()
    try:
        query = run_select(nl2sql_result.sql)
    except QueryError as exc:
        return Answer(
            ok=False,
            question=question,
            sql=nl2sql_result.sql,
            error=str(exc),
            nl2sql=nl2sql_result,
            seconds=time.time() - started,
            timings=timings,
        )
    timings["query"] = round(time.time() - t0, 2)

    t0 = time.time()
    analysis = analyze(query.df, question, layer)
    timings["analysis"] = round(time.time() - t0, 2)

    t0 = time.time()
    if with_narrative:
        narrative = generate_narrative(analysis, question, model=model)
    else:
        narrative = Narrative(headline=question, summary="", model="disabled")
    timings["narrative"] = round(time.time() - t0, 2)

    spec = pick_chart(query.df, analysis.time_col, analysis.dimension_col)
    chart_png = None
    if with_chart and query.df is not None and not query.df.empty:
        t0 = time.time()
        try:
            chart_png = render_png(query.df, spec, show_title=False)
        except Exception:
            chart_png = None
        timings["chart"] = round(time.time() - t0, 2)

    bundle = ReportBundle(
        question=question,
        sql=nl2sql_result.sql,
        df=query.df,
        analysis=analysis,
        narrative=narrative,
        chart_png=chart_png,
        chart_spec=spec.to_dict(),
        subtitle=f"{query.rows} rows in {query.seconds * 1000:.0f} ms",
        meta={
            "query_seconds": query.seconds,
            "model": narrative.model,
            "truncated": query.truncated,
        },
    )

    return Answer(
        ok=True,
        question=question,
        sql=nl2sql_result.sql,
        bundle=bundle,
        nl2sql=nl2sql_result,
        seconds=time.time() - started,
        timings=timings,
    )


def export(bundle: ReportBundle, formats: tuple[str, ...] = ("pdf",), out_dir: str | Path | None = None) -> dict[str, str]:
    directory = Path(out_dir or REPORT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{slugify(bundle.question)}"

    written: dict[str, str] = {}
    for kind in formats:
        builder = BUILDERS.get(kind)
        if builder is None:
            continue
        target = directory / f"{base}.{kind}"
        builder(bundle, target)
        written[kind] = str(target)

    return written


def generate_report(
    question: str,
    formats: tuple[str, ...] = ("pdf",),
    out_dir: str | Path | None = None,
    model: str | None = None,
) -> dict:
    answer = answer_question(question, model=model)
    if not answer.ok:
        return answer.to_dict()

    files = export(answer.bundle, formats=formats, out_dir=out_dir)
    payload = answer.to_dict()
    payload["files"] = files
    return payload
