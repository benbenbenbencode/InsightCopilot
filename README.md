# InsightCopilot

Ask a business question in plain English. Get a validated SQL query, an
analysis with real numbers, a written narrative, an interactive chart, and a
shareable **PDF / PowerPoint / Excel** report — running entirely on your own
machine with a local model.

![status](https://img.shields.io/badge/LLM-local%20(Ollama)-6c5ce7)

---

## What changed in v2

The v1 pipeline looked like NL→SQL but wasn't: the model's SQL was thrown away
after table extraction, and the executed query was rebuilt from keyword
matching (`"by city"`, `"by product"`). Anything phrased differently silently
fell back to a fixed query. The "root cause" and "recommendations" panels were
hardcoded string templates in the chart module.

| v1 | v2 |
|---|---|
| LLM SQL discarded, query rebuilt from keywords | model's SQL is executed, after whitelist + `EXPLAIN` validation and a bounded repair loop |
| `subprocess.run(["ollama", "run", ...])` per request | one HTTP client, retries, timing, JSON-schema output |
| `deepseek-coder` (1B) | `qwen3.6` (36B MoE) with a measured fallback |
| recommendations were f-strings | findings computed from data; narrative generated, then number-checked |
| no exports | PDF, PPTX, XLSX, plus an interactive dashboard |
| schema scattered across 4 modules | one `semantic/semantic_layer.yml` |

## Architecture

```
question
   |
   v
[semantic layer]  semantic/semantic_layer.yml   (tables, metrics, joins, synonyms, few-shots)
   |
   v
[prompt]  tables + metrics + coverage + examples
   |
   v
[local model]  llm/client.py  ->  Ollama /api/chat
   |
   v
[validate]  core/nl2sql.py    static whitelist  ->  EXPLAIN against PostgreSQL  ->  repair & retry
   |
   v
[execute]   core/executor.py  read-only txn, 30s timeout, row cap
   |
   v
[analyse]   core/analysis.py  KPIs, trend, concentration, MAD anomalies
   |
   v
[narrate]   core/narrative.py structured JSON + numeric grounding check
   |
   +--> dashboard (Streamlit)  app/dashboard.py
   +--> API (FastAPI)          api/main.py
   +--> reports                report/  ->  PDF | PPTX | XLSX
```

## The semantic layer

`semantic/semantic_layer.yml` is the contract. Adding a metric is a YAML edit,
not a code change:

```yaml
metrics:
  - id: revenue_per_user
    label: Revenue per user
    sql: SUM(fct_transactions.revenue) / NULLIF(COUNT(DISTINCT fct_transactions.user_id), 0)
    format: currency
    synonyms: [arpu, revenue per user, pendapatan per pelanggan]
```

It drives three things at once: the prompt (so the model uses your business
definitions verbatim), the validator (allowed tables), and the report renderer
(metric labels and formats). Synonyms are bilingual — "omzet", "kota",
"pelanggan" resolve the same as "revenue", "city", "customer".

## Model choice (measured, not guessed)

Benchmarked on this machine — RTX 5080 16 GB, 200k-row retail dataset, two
NL→SQL probe questions:

| Model | Size | Warm throughput | Notes |
|---|---|---|---|
| **qwen3.6:latest** (36B MoE) | 22.3 GB | **60–65 tok/s** | both queries correct → **default** |
| qwen2.5-coder:7b | 4.4 GB | 150–160 tok/s | fastest, but misread "revenue per user" as avg per row |
| Qwen3.8-27B Q4 | 15.7 GB | 15 tok/s | correct, noticeably slower |

The 36B is larger than VRAM, but it is an MoE — only a fraction of parameters
are active per token, so CPU offload still yields 60+ tok/s. First call after
idle takes ~40 s to load; `keep_alive=30m` avoids paying that repeatedly.

Set `SQL_MODEL` / `FAST_MODEL` in `.env` to switch.

## Reports

Every answer can be exported; the same `ReportBundle` feeds all three formats,
so what you see on screen is what lands in the file.

- **PDF** — cover, executive summary, KPI cards, chart, findings, actions,
  caveats, data table, full SQL in a provenance section
- **PPTX** — six slides, 16:9, matching theme
- **XLSX** — Summary, KPIs, Findings, Actions, Data sheets, styled and frozen

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env          # set DB_PASSWORD and model names
ollama pull qwen3.6           # or point SQL_MODEL at any installed model
```

```bash
streamlit run streamlit_app.py        # dashboard
python -m uvicorn api.main:app --reload   # API on :8000, docs at /docs
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | database, Ollama, semantic layer status |
| `GET /semantic` | the semantic layer as JSON |
| `POST /sql` | question → validated SQL, nothing executed |
| `POST /ask` | question → SQL, rows, analysis, narrative |
| `POST /report` | question → generated files (pdf/pptx/xlsx) |
| `GET /download/{file}` | fetch a generated report |

## Verification

```bash
python -m scripts.smoke_nl2sql        # 5 questions through the real pipeline
python -m scripts.smoke_report        # full report build, writes to reports/
python -m scripts.test_dashboard      # Streamlit render + one interaction, headless
python -m pytest tests/ -q
```

Current state: 5/5 NL→SQL questions correct on first attempt (1–11 s each),
reports generate in ~8 s, dashboard renders without exceptions, narrative
grounding passes.

## Fine-tuning

`training/` builds a dataset from the semantic layer and trains a QLoRA adapter:

```bash
python -m training.build_dataset --count 200
python -m training.train_qlora --model Qwen/Qwen2.5-Coder-7B-Instruct
ollama create retail-sql -f training/Modelfile
```

See [training/README.md](training/README.md). Be realistic: a 36B cannot be
trained on 16 GB. Train a 7B adapter for style/refusals and keep the 36B for
hard questions. The semantic layer and few-shot examples usually buy more than
a fine-tune.

## Limitations

- **Scoped to one schema by design.** This is not a general text-to-SQL tool.
- **30 days of data** (2026-02-18 → 2026-03-19). The prompt says so explicitly,
  so relative questions anchor to the data, not `CURRENT_DATE`.
- **Grounding check is a heuristic** — numbers are compared within 2% of
  computed values. It catches fabricated figures, not misinterpreted ones.
- **The narrative can still be wrong** about *why* something happened; it only
  describes what the data shows.
- **Fine-tuning is untested on this machine** (no CUDA training stack
  installed). The scripts are standard PEFT/transformers; WSL2 recommended.
- Legacy v1 modules (`core/pipeline.py`, `app/chart.py`, `llm/deepseek_sql.py`,
  `core/insight.py`) are left in place but unused. They still contain the
  keyword builder; nothing imports them.
