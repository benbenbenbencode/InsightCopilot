"""Shared report model passed between builder, PDF, PPTX and XLSX renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from core.analysis import Analysis
from core.narrative import Narrative


@dataclass
class ReportBundle:
    question: str
    sql: str
    df: pd.DataFrame
    analysis: Analysis
    narrative: Narrative
    chart_png: bytes | None = None
    chart_spec: dict = field(default_factory=dict)
    title: str = "Retail Insight Report"
    subtitle: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    meta: dict = field(default_factory=dict)

    @property
    def stamp(self) -> str:
        return self.generated_at.strftime("%d %b %Y, %H:%M")
