"""InsightCopilot dashboard.

One page, three tabs: ask a question, inspect the semantic layer, and see what
the local model stack is doing. Every result can be exported as PDF, PPTX or
XLSX without leaving the page.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core.executor import health as db_health
from llm.client import get_client
from report.builder import REPORT_DIR, answer_question, export
from semantic import load as load_semantic_layer

ACCENT = "#6c5ce7"

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1250px; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  .ic-kpi {{
      background: #f8f8fc; border: 1px solid #e6e6ee; border-radius: 12px;
      padding: 14px 16px; height: 100%;
  }}
  .ic-kpi .label {{ font-size: 0.72rem; letter-spacing: .08em; text-transform: uppercase; color: #7a7a88; }}
  .ic-kpi .value {{ font-size: 1.65rem; font-weight: 650; color: #181820; line-height: 1.25; }}
  .ic-chip {{
      display: inline-block; font-size: .72rem; padding: 2px 9px; border-radius: 999px;
      background: #eeebff; color: {ACCENT}; font-weight: 600; margin-right: 6px;
  }}
  .ic-sev {{ font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
  .ic-sev.info {{ color: {ACCENT}; }}
  .ic-sev.positive {{ color: #009678; }}
  .ic-sev.warning {{ color: #c47814; }}
  .ic-sev.critical {{ color: #be3232; }}
  .ic-sql {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: .8rem; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
</style>
"""

EXAMPLES = [
    "What is total revenue by product category, and which one is underperforming?",
    "How did daily revenue trend over the last 30 days?",
    "Which cities generate the most revenue per user?",
    "What is the average order value per product?",
    "Which products have the best ROAS?",
    "How many distinct users bought something in each city?",
]


# ---------------------------------------------------------------------------
# cached probes
# ---------------------------------------------------------------------------


@st.cache_data(ttl=15, show_spinner=False)
def probe() -> dict:
    client = get_client()
    return {
        "db": db_health(),
        "ollama_up": client.is_up(),
        "models": [m.get("name", "") for m in client.models()],
        "host": client.base_url,
    }


def kpi_cards(kpis) -> None:
    if not kpis:
        return
    columns = st.columns(min(4, len(kpis)))
    for column, kpi in zip(columns, kpis[:4]):
        with column:
            st.markdown(
                f"<div class='ic-kpi'><div class='label'>{kpi.label}</div>"
                f"<div class='value'>{kpi.display}</div></div>",
                unsafe_allow_html=True,
            )


def severity_badge(severity: str) -> str:
    return f"<span class='ic-sev {severity}'>{severity}</span>"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def render_answer(answer, model: str) -> None:
    if not answer.ok:
        st.error(f"**Could not answer that.** {answer.error}")
        if answer.sql:
            st.code(answer.sql, language="sql")
        if answer.nl2sql and answer.nl2sql.attempts:
            with st.expander("Attempts"):
                for attempt in answer.nl2sql.attempts:
                    st.markdown(f"**{attempt.stage}** — {attempt.error}")
                    st.code(attempt.sql or "(none)", language="sql")
        return

    bundle = answer.bundle
    analysis = bundle.analysis
    narrative = bundle.narrative

    if narrative.headline:
        st.subheader(narrative.headline)
    if narrative.summary:
        st.markdown(narrative.summary)

    kpi_cards(analysis.kpis)

    left, right = st.columns([2, 1], gap="large")

    with left:
        if not bundle.df.empty:
            from report.charts import pick_chart, render_altair

            spec = pick_chart(bundle.df, analysis.time_col, analysis.dimension_col)
            chart = render_altair(bundle.df, spec)
            if chart is not None:
                st.altair_chart(chart)
        else:
            st.info("The query returned no rows.")

    with right:
        st.markdown("**Key points**")
        if narrative.key_points:
            for point in narrative.key_points:
                st.markdown(f"- {point}")
        else:
            st.caption("No narrative available.")

        if narrative.recommendations:
            st.markdown("**Recommended actions**")
            for index, rec in enumerate(narrative.recommendations, 1):
                st.markdown(f"{index}. {rec}")

    if analysis.findings:
        st.markdown("**Findings**")
        for finding in analysis.findings:
            st.markdown(
                f"{severity_badge(finding.severity)} &nbsp; **{finding.title}**  \n"
                f"<span style='color:#6a6a78'>{finding.detail}</span>",
                unsafe_allow_html=True,
            )

    if narrative.risks:
        with st.expander("Caveats", expanded=False):
            for risk in narrative.risks:
                st.warning(risk, icon="⚠️")

    if not narrative.grounded:
        st.warning(
            "Some figures in the summary could not be traced back to the computed results: "
            + ", ".join(f"{n:,.0f}" for n in narrative.ungrounded_numbers)
        )

    st.divider()

    tab_data, tab_sql, tab_export = st.tabs(["Data", "SQL", "Export"])

    with tab_data:
        st.dataframe(bundle.df, height=340)
        st.caption(
            f"{len(bundle.df)} rows · query {bundle.meta.get('query_seconds', 0) * 1000:.0f} ms · "
            f"model {narrative.model} · {answer.seconds:.1f}s total"
        )

    with tab_sql:
        st.code(bundle.sql, language="sql")
        if answer.nl2sql and answer.nl2sql.repaired:
            st.caption(f"Repaired after {len(answer.nl2sql.attempts) - 1} failed attempt(s).")

    with tab_export:
        st.caption("Reports are rendered locally from this exact result.")
        columns = st.columns(3)
        for column, (kind, label) in zip(
            columns, (("pdf", "PDF report"), ("pptx", "PowerPoint deck"), ("xlsx", "Excel workbook"))
        ):
            with column:
                if st.button(label, key=f"export-{kind}"):
                    with st.spinner(f"Rendering {kind}…"):
                        files = export(bundle, formats=(kind,))
                    path = files.get(kind)
                    if path and Path(path).exists():
                        st.session_state[f"download-{kind}"] = path

                stored = st.session_state.get(f"download-{kind}")
                if stored and Path(stored).exists():
                    st.download_button(
                        f"Download {kind.upper()}",
                        data=Path(stored).read_bytes(),
                        file_name=Path(stored).name,
                        mime={
                            "pdf": "application/pdf",
                            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        }[kind],
                        key=f"dl-{kind}",
                    )


def tab_ask(model: str) -> None:
    question = st.text_area(
        "Ask a business question",
        value=st.session_state.get("question", ""),
        placeholder=EXAMPLES[0],
        height=88,
        key="question_input",
    )

    columns = st.columns([1, 1, 4])
    with columns[0]:
        run = st.button("Analyse", type="primary", use_container_width=True)
    with columns[1]:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop("answer", None)
        st.rerun()

    st.caption("Examples")
    example_columns = st.columns(3)
    for index, example in enumerate(EXAMPLES):
        with example_columns[index % 3]:
            if st.button(example, key=f"example-{index}", use_container_width=True):
                st.session_state["pending_question"] = example
                st.rerun()

    if st.session_state.get("pending_question"):
        pending = st.session_state.pop("pending_question")
        with st.spinner("Thinking locally…"):
            st.session_state["answer"] = answer_question(pending, model=model)
        st.session_state["question"] = pending

    if run and question.strip():
        with st.spinner("Thinking locally…"):
            st.session_state["answer"] = answer_question(question, model=model)

    answer = st.session_state.get("answer")
    if answer:
        st.divider()
        render_answer(answer, model)
    else:
        st.info("Ask a question, or pick one of the examples above.")


def tab_semantic() -> None:
    layer = load_semantic_layer()
    st.caption(
        f"{layer.name} · {len(layer.metrics)} metrics · {len(layer.dimensions)} dimensions · "
        f"{len(layer.entities)} tables · source: {layer.source_path.name}"
    )

    st.markdown("**Entities**")
    for entity in layer.entities.values():
        with st.expander(f"{entity.name} — {entity.label}"):
            if entity.description:
                st.caption(entity.description)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"column": name, "type": meta.get("type", ""), "description": meta.get("description", "")}
                        for name, meta in entity.columns.items()
                    ]
                ),
                hide_index=True,
            )

    st.markdown("**Metrics**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": metric.id,
                    "label": metric.label,
                    "sql": metric.sql,
                    "aliases": ", ".join(metric.synonyms),
                }
                for metric in layer.metrics.values()
            ]
        ),
        hide_index=True,
    )

    st.markdown("**Dimensions**")
    st.dataframe(
        pd.DataFrame(
            [
                {"dimension": dim.id, "label": dim.label, "sql": dim.sql, "aliases": ", ".join(dim.synonyms)}
                for dim in layer.dimensions.values()
            ]
        ),
        hide_index=True,
    )

    st.markdown("**Join paths**")
    st.dataframe(
        pd.DataFrame([{"condition": join.on, "type": join.type} for join in layer.joins]),
        hide_index=True,
    )


def tab_model(info: dict, model: str) -> None:
    left, right = st.columns(2)

    with left:
        st.markdown("**Local stack**")
        db = info.get("db", {})
        st.write(f"Database: {'connected' if db.get('ok') else 'unavailable'} — {db.get('database', '')}")
        st.write(f"Ollama: {'running' if info.get('ollama_up') else 'not running'} at {info.get('host')}")
        st.write(f"Active model: `{model}`")
        st.write(f"Installed models: {len(info.get('models', []))}")

        if info.get("models"):
            st.dataframe(pd.DataFrame({"model": info["models"]}), hide_index=True, height=260)

    with right:
        st.markdown("**Training**")
        st.caption(
            "The fine-tuning pipeline lives in `training/`. It builds a dataset from the "
            "semantic layer and trains a QLoRA adapter, which you then import into Ollama."
        )
        dataset_path = Path("training/data/train.jsonl")
        if dataset_path.exists():
            lines = sum(1 for _ in dataset_path.open("r", encoding="utf-8"))
            st.success(f"Dataset ready: {dataset_path} ({lines} examples)")
        else:
            st.info("No dataset yet. Run: python -m training.build_dataset")

        st.code(
            "python -m training.build_dataset --count 300\n"
            "python -m training.train_qlora --model Qwen/Qwen2.5-Coder-7B-Instruct\n"
            "ollama create retail-sql -f training/Modelfile",
            language="bash",
        )


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="InsightCopilot", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    info = probe()

    defaults = [
        os.environ.get("SQL_MODEL", "qwen3.6:latest"),
        os.environ.get("FAST_MODEL", "qwen2.5-coder:7b"),
    ]
    options = list(dict.fromkeys([m for m in defaults if m] + info.get("models", [])))

    with st.sidebar:
        st.title("InsightCopilot")
        st.caption("Local-first retail analytics")

        model = st.selectbox("Model", options=options, index=0)

        status_colour = "🟢" if info.get("ollama_up") else "🔴"
        st.write(f"{status_colour} Ollama")
        db = info.get("db", {})
        st.write(("🟢" if db.get("ok") else "🔴") + " PostgreSQL")
        if db.get("ok"):
            st.caption(f"{db.get('database')} · {db.get('version', '')[:28]}")

        st.divider()
        st.caption(
            "Every query runs read-only with a 30s timeout and a row cap. "
            "Nothing leaves this machine."
        )

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("answer", None)
            st.rerun()

    st.title("📊 InsightCopilot")
    st.caption("Ask in plain English. Get SQL, analysis, a narrative, and a shareable report — all locally.")

    tab1, tab2, tab3 = st.tabs(["Ask", "Semantic layer", "Model & training"])

    with tab1:
        tab_ask(model)
    with tab2:
        tab_semantic()
    with tab3:
        tab_model(info, model)


if __name__ == "__main__":
    main()
