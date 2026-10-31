"""Narrative generation: the only place an LLM writes prose.

The model receives already-computed findings and is asked for structured JSON
(headline / summary / key points / recommendations / risks). Numbers are
cross-checked afterwards, so a fabricated figure is flagged rather than
presented as fact.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from core.analysis import Analysis, fmt_number
from llm.client import LLMError, OllamaClient, get_client

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "summary", "key_points", "recommendations", "risks"],
}

SYSTEM = """You are a retail analytics lead writing for an executive audience.
You are given findings that have ALREADY been computed in Python.

Hard rules:
- Use ONLY the numbers present in the findings. Never invent, extrapolate or round to new values.
- Do not recompute totals or percentages.
- Be specific and direct. No filler, no "in today's fast-paced world".
- summary: 2-4 sentences describing what the data shows and why it matters.
- key_points: 3-5 bullets, each one insight (not a restatement of a column name).
- recommendations: 2-4 concrete actions, each tied to a finding.
- risks: 1-3 caveats about the data or the conclusion. Empty list if none.
- Write in the same language as the question.
"""


@dataclass
class Narrative:
    headline: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    grounded: bool = True
    ungrounded_numbers: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "key_points": self.key_points,
            "recommendations": self.recommendations,
            "risks": self.risks,
            "model": self.model,
            "seconds": round(self.seconds, 2),
            "grounded": self.grounded,
            "ungrounded_numbers": self.ungrounded_numbers,
        }


NUMBER_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([KkMmBb])?")
MULTIPLIERS = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}

# Small integers that appear as prose ("3 key points", "2x growth"), not as data.
TRIVIAL_NUMBERS = {0, 1, 2, 3, 4, 5, 6, 10, 12, 24, 80, 100}


def _numbers(text: str) -> set[float]:
    """Extract numbers, honouring K/M/B suffixes so '199.6M' means 199,600,000."""
    found: set[float] = set()
    for token, suffix in NUMBER_RE.findall(text or ""):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if suffix:
            value *= MULTIPLIERS[suffix.lower()]
        found.add(value)
    return found


def _known_numbers(analysis: Analysis) -> set[float]:
    """Every number the analysis has legitimately produced."""
    known: set[float] = {float(analysis.rows)}

    for kpi in analysis.kpis:
        known.add(float(kpi.value))

    for finding in analysis.findings:
        if finding.value is not None:
            known.add(float(finding.value))
        for value in finding.evidence.values():
            if isinstance(value, (int, float)):
                known.add(float(value))

    if analysis.breakdown is not None and not analysis.breakdown.empty:
        metric = analysis.breakdown.columns[-1]
        values: list[float] = []
        for value in analysis.breakdown[metric].head(20):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        known.update(values)

        # Narratives naturally quote each group's share of the total, and the
        # gap between the leader and a laggard. Compute both here so a correct
        # derived figure is not flagged as fabricated.
        total = sum(values)
        if total:
            known.update(v / total * 100 for v in values)

        head = values[:8]
        for i, left in enumerate(head):
            for right in head[i + 1 :]:
                gap = abs(left - right)
                known.add(gap)
                if total:
                    known.add(gap / total * 100)

    return known


def _is_grounded(claimed: float, known: set[float], tolerance: float = 0.02) -> bool:
    """A figure is grounded if it is within 2% of a computed value.

    The tolerance matters because narratives round ("199.6M" for 199,581,611)
    and because percentages and ratios are quoted at one decimal place.
    """
    if claimed in TRIVIAL_NUMBERS:
        return True

    for value in known:
        if value == 0:
            if abs(claimed) < 1:
                return True
            continue
        if abs(claimed - value) <= max(abs(value) * tolerance, 0.51):
            return True

    return False


def _fallback(analysis: Analysis, question: str) -> Narrative:
    """Used when the model fails -- never leave the report empty."""
    headline = f"Results for: {question}" if question else "Analysis results"
    summary_parts = []
    if analysis.kpis:
        summary_parts.append(
            "Key figures: "
            + ", ".join(f"{k.label} {k.display}" for k in analysis.kpis[:4])
            + "."
        )
    summary_parts.append(f"{analysis.rows} row(s) returned.")

    return Narrative(
        headline=headline,
        summary=" ".join(summary_parts),
        key_points=[f.title for f in analysis.findings][:5],
        recommendations=[],
        risks=["Narrative generation was unavailable; this report shows computed results only."],
        model="fallback",
    )


def generate_narrative(
    analysis: Analysis,
    question: str,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> Narrative:
    if analysis.rows == 0:
        return Narrative(
            headline="No data matched the question",
            summary="The query executed successfully but returned no rows. The filters may be too narrow, or the period has no activity.",
            key_points=[],
            recommendations=["Widen the date range or remove a filter and re-run."],
            risks=["An empty result is not proof of zero activity unless the filters are known to be correct."],
            model="rule-based",
        )

    client = client or get_client()
    model = model or os.environ.get("INSIGHT_MODEL") or os.environ.get("SQL_MODEL") or "qwen3.6:latest"

    prompt = f"""Question: {question}

Computed findings:
{analysis.narrative_context()}

Write the executive JSON report now."""

    try:
        result = client.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.2,
            num_ctx=8192,
            num_predict=900,
            json_schema=NARRATIVE_SCHEMA,
        )
    except LLMError as exc:
        fallback = _fallback(analysis, question)
        fallback.risks.append(f"Model error: {exc}")
        return fallback

    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError:
        start = result.text.find("{")
        end = result.text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(result.text[start : end + 1])
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None

    if not isinstance(payload, dict):
        fallback = _fallback(analysis, question)
        fallback.risks.append("The model returned malformed JSON.")
        return fallback

    narrative = Narrative(
        headline=str(payload.get("headline", "")).strip(),
        summary=str(payload.get("summary", "")).strip(),
        key_points=[str(x).strip() for x in payload.get("key_points") or [] if str(x).strip()],
        recommendations=[str(x).strip() for x in payload.get("recommendations") or [] if str(x).strip()],
        risks=[str(x).strip() for x in payload.get("risks") or [] if str(x).strip()],
        model=result.model,
        seconds=result.seconds,
    )

    # Grounding check: flag figures the analysis cannot support.
    known = _known_numbers(analysis)
    text = " ".join([narrative.headline, narrative.summary, *narrative.key_points])
    suspicious = sorted(number for number in _numbers(text) if not _is_grounded(number, known))

    if suspicious:
        narrative.grounded = False
        narrative.ungrounded_numbers = [float(n) for n in suspicious[:8]]
        narrative.risks.append(
            "Some figures in the narrative could not be traced to the computed results: "
            + ", ".join(fmt_number(n) for n in suspicious[:5])
        )

    return narrative
