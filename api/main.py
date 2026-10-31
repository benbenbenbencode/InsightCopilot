"""InsightCopilot API.

Endpoints
  GET  /health                 database + model status
  GET  /semantic               the semantic layer as JSON
  POST /sql                    question -> validated SQL (no execution)
  POST /ask                    question -> analysis + narrative + rows
  POST /report                 question -> generated files (pdf/pptx/xlsx)
  GET  /download/{filename}    fetch a generated report
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.executor import health as db_health
from core.nl2sql import question_to_sql
from llm.client import get_client
from report.builder import REPORT_DIR, answer_question, export
from semantic import load as load_semantic_layer

app = FastAPI(
    title="InsightCopilot API",
    description="Local-first retail analytics: NL->SQL, analysis, narrative and reports.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Business question in plain language")
    model: str | None = Field(default=None, description="Override the Ollama model")
    with_narrative: bool = Field(default=True, description="Generate the LLM narrative")


class ReportRequest(QuestionRequest):
    formats: list[Literal["pdf", "pptx", "xlsx"]] = Field(default=["pdf"])


@app.get("/")
def root():
    return {
        "service": "InsightCopilot",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/semantic", "/sql", "/ask", "/report"],
    }


@app.get("/health")
def health():
    client = get_client()
    database = db_health()
    layer = None
    try:
        loaded = load_semantic_layer()
        layer = {
            "name": loaded.name,
            "metrics": len(loaded.metrics),
            "dimensions": len(loaded.dimensions),
            "tables": len(loaded.entities),
        }
    except Exception as exc:
        layer = {"error": str(exc)}

    return {
        "database": database,
        "ollama": {
            "up": client.is_up(),
            "host": client.base_url,
            "models": [m.get("name") for m in client.models()],
        },
        "semantic_layer": layer,
        "sql_model": os.environ.get("SQL_MODEL"),
    }


@app.get("/semantic")
def semantic():
    layer = load_semantic_layer()
    return {
        "name": layer.name,
        "description": layer.description,
        "entities": {
            name: {
                "label": entity.label,
                "grain": entity.grain,
                "columns": entity.columns,
            }
            for name, entity in layer.entities.items()
        },
        "metrics": [
            {"id": m.id, "label": m.label, "sql": m.sql, "format": m.format, "synonyms": m.synonyms}
            for m in layer.metrics.values()
        ],
        "dimensions": [
            {"id": d.id, "label": d.label, "sql": d.sql, "type": d.type, "synonyms": d.synonyms}
            for d in layer.dimensions.values()
        ],
        "joins": [{"left": j.left, "right": j.right, "on": j.on} for j in layer.joins],
        "examples": layer.examples,
    }


@app.post("/sql")
def sql_only(request: QuestionRequest):
    result = question_to_sql(request.question, model=request.model)
    return result.to_dict()


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = answer_question(
        request.question,
        model=request.model,
        with_narrative=request.with_narrative,
    )
    return answer.to_dict()


@app.post("/report")
def report(request: ReportRequest):
    answer = answer_question(request.question, model=request.model)
    if not answer.ok:
        return answer.to_dict(max_rows=0)

    files = export(answer.bundle, formats=tuple(request.formats))
    payload = answer.to_dict()
    payload["files"] = {
        kind: {"path": path, "filename": Path(path).name, "url": f"/download/{Path(path).name}"}
        for kind, path in files.items()
    }
    return payload


@app.get("/download/{filename}")
def download(filename: str):
    # Reject path traversal: only serve files that live directly in REPORT_DIR.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = (Path(REPORT_DIR) / filename).resolve()
    if target.parent != Path(REPORT_DIR).resolve() or not target.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(target, filename=target.name)
