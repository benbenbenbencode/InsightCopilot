"""Deterministic analysis over query results.

The rule is: maths happens in Python, wording happens in the model. Every
number a narrative quotes must already exist in an `Analysis` produced here,
so the LLM is describing findings rather than inventing them.

Outputs are deliberately plain data structures (dataclasses / dicts) so the
same object can drive the API response, the PDF, the PPTX and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MAD_THRESHOLD = 3.5


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def fmt_number(value: Any, digits: int = 1) -> str:
    """Compact, human-readable number: 1.24M, 15.3K, 3.10."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return "-"

    sign = "-" if number < 0 else ""
    number = abs(number)

    if number >= 1_000_000_000:
        return f"{sign}{number / 1_000_000_000:.{digits}f}B"
    if number >= 1_000_000:
        return f"{sign}{number / 1_000_000:.{digits}f}M"
    if number >= 10_000:
        return f"{sign}{number / 1_000:.{digits}f}K"
    if number >= 100:
        return f"{sign}{number:,.0f}"
    if number == int(number):
        return f"{sign}{int(number)}"
    return f"{sign}{number:,.2f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}%"


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    kind: str
    title: str
    detail: str
    severity: str = "info"  # info | positive | warning | critical
    value: float | None = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "severity": self.severity,
            "value": self.value,
            "evidence": self.evidence,
        }


@dataclass
class KPI:
    label: str
    value: float
    display: str
    format: str = "number"
    column: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "display": self.display,
            "format": self.format,
            "column": self.column,
        }


@dataclass
class Analysis:
    rows: int
    columns: list[str]
    kpis: list[KPI]
    findings: list[Finding]
    metric_cols: list[str]
    time_col: str | None = None
    dimension_col: str | None = None
    timeseries: pd.DataFrame | None = None
    breakdown: pd.DataFrame | None = None
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "kpis": [k.to_dict() for k in self.kpis],
            "findings": [f.to_dict() for f in self.findings],
            "metric_cols": self.metric_cols,
            "time_col": self.time_col,
            "dimension_col": self.dimension_col,
            "quality": self.quality,
        }

    def narrative_context(self) -> str:
        """Compact, factual block for the model to write prose from."""
        lines = [f"Rows returned: {self.rows}", f"Columns: {', '.join(self.columns)}"]

        if self.kpis:
            lines.append("\nKPIs:")
            for kpi in self.kpis:
                lines.append(f"- {kpi.label}: {kpi.display}")

        if self.findings:
            lines.append("\nFindings (already computed -- do not recompute):")
            for idx, finding in enumerate(self.findings, 1):
                lines.append(f"{idx}. [{finding.severity}] {finding.title} -- {finding.detail}")

        if self.timeseries is not None and not self.timeseries.empty:
            head = self.timeseries.head(3).to_string(index=False)
            tail = self.timeseries.tail(3).to_string(index=False)
            lines.append(f"\nTime series head:\n{head}\n... tail:\n{tail}")

        if self.breakdown is not None and not self.breakdown.empty:
            lines.append(f"\nBreakdown by {self.dimension_col} (top rows):")
            lines.append(self.breakdown.head(8).to_string(index=False))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# column detection
# ---------------------------------------------------------------------------

META_NAME_HINTS = ("id", "key", "uuid", "code")
TIME_NAME_HINTS = ("date", "month", "week", "day", "period", "time", "year")
DIMENSION_NAME_HINTS = ("city", "product", "category", "name", "segment", "country", "store", "channel")


def _is_time_like(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(hint in name.lower() for hint in TIME_NAME_HINTS):
        try:
            pd.to_datetime(series.dropna().head(20))
            return True
        except (ValueError, TypeError):
            return False
    return False


def _pick_columns(df: pd.DataFrame) -> tuple[list[str], str | None, str | None]:
    metric_cols = list(df.select_dtypes(include="number").columns)

    time_col = None
    for column in df.columns:
        if column in metric_cols:
            continue
        if _is_time_like(df[column], column):
            time_col = column
            break

    dimension_col = None
    for column in df.columns:
        if column in metric_cols or column == time_col:
            continue
        lowered = column.lower()
        if any(hint in lowered for hint in META_NAME_HINTS):
            continue
        if df[column].nunique(dropna=True) > 1:
            dimension_col = column
            break

    if dimension_col is None:
        for column in df.columns:
            if column in metric_cols or column == time_col:
                continue
            if df[column].nunique(dropna=True) > 1:
                dimension_col = column
                break

    return metric_cols, time_col, dimension_col


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------


def _kpis(df: pd.DataFrame, metric_cols: list[str], layer=None) -> list[KPI]:
    kpis: list[KPI] = []
    metric_labels = {}
    if layer is not None:
        metric_labels = {m.id: m for m in layer.metrics.values()}
        by_sql = {}
        for metric in layer.metrics.values():
            by_sql[metric.id] = metric

    for column in metric_cols[:4]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        lowered = column.lower()
        is_currency = any(token in lowered for token in ("revenue", "sales", "cost", "spend", "value", "price"))
        is_ratio = any(token in lowered for token in ("roas", "roi", "rate", "ratio", "pct", "percent"))
        is_average = lowered.startswith(("avg", "average", "mean", "arpu", "aov"))

        label = column.replace("_", " ").title()
        if layer is not None and column in metric_labels:
            label = metric_labels[column].label

        if is_ratio:
            value = float(series.mean())
            display = f"{value:.2f}x"
            fmt = "ratio"
        elif is_average:
            value = float(series.mean())
            display = fmt_number(value)
            fmt = "currency" if is_currency else "number"
        else:
            value = float(series.sum())
            display = fmt_number(value)
            fmt = "currency" if is_currency else "number"

        kpis.append(KPI(label=label, value=value, display=display, format=fmt, column=column))

    if df.shape[0] and not metric_cols:
        kpis.append(KPI(label="Rows", value=float(len(df)), display=f"{len(df):,}", format="integer"))

    return kpis


def _trend_findings(df: pd.DataFrame, time_col: str, metric: str) -> tuple[pd.DataFrame, list[Finding]]:
    working = df[[time_col, metric]].copy()
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna().sort_values(time_col)

    if len(working) < 3:
        return working, []

    findings: list[Finding] = []
    first, last = working.iloc[0], working.iloc[-1]
    change = ((last[metric] - first[metric]) / abs(first[metric]) * 100) if first[metric] else 0.0

    peak_idx = working[metric].idxmax()
    trough_idx = working[metric].idxmin()
    peak = working.loc[peak_idx]
    trough = working.loc[trough_idx]

    severity = "positive" if change > 5 else "warning" if change < -5 else "info"
    findings.append(
        Finding(
            kind="trend",
            title=f"{metric.replace('_', ' ').title()} moved {fmt_pct(change)} across the period",
            detail=(
                f"From {fmt_number(first[metric])} in the first period to "
                f"{fmt_number(last[metric])} in the last. Peak {fmt_number(peak[metric])} "
                f"({pd.Timestamp(peak[time_col]).date()}), low {fmt_number(trough[metric])} "
                f"({pd.Timestamp(trough[time_col]).date()})."
            ),
            severity=severity,
            value=float(change),
            evidence={
                "first": float(first[metric]),
                "last": float(last[metric]),
                "change_pct": float(change),
                "peak": str(pd.Timestamp(peak[time_col]).date()),
                "trough": str(pd.Timestamp(trough[time_col]).date()),
            },
        )
    )

    # MAD-based anomalies: robust to a single extreme spike, unlike z-score.
    values = working[metric].to_numpy(dtype=float)
    median = np.median(values)
    deviations = np.abs(values - median)
    mad = np.median(deviations)

    if mad > 0:
        scores = 0.6745 * (values - median) / mad
        outliers = working[np.abs(scores) > MAD_THRESHOLD]
        if not outliers.empty:
            worst_index = int(np.argmax(np.abs(scores)))
            worst = working.iloc[worst_index]
            findings.append(
                Finding(
                    kind="anomaly",
                    title=f"{len(outliers)} anomalous period(s) detected",
                    detail=(
                        f"{pd.Timestamp(worst[time_col]).date()} deviates most "
                        f"({fmt_number(worst[metric])} vs median {fmt_number(median)})."
                    ),
                    severity="warning",
                    value=float(scores[worst_index]),
                    evidence={
                        "dates": [str(pd.Timestamp(d).date()) for d in outliers[time_col].head(5)],
                        "median": float(median),
                        "mad": float(mad),
                    },
                )
            )

    return working, findings


def _breakdown_findings(
    df: pd.DataFrame, dimension: str, metric: str, total: float
) -> tuple[pd.DataFrame, list[Finding]]:
    working = df[[dimension, metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna().groupby(dimension, as_index=False)[metric].sum()
    working = working.sort_values(metric, ascending=False)

    if working.empty:
        return working, []

    findings: list[Finding] = []
    grand_total = working[metric].sum()
    top = working.iloc[0]
    bottom = working.iloc[-1]

    top_share = (top[metric] / grand_total * 100) if grand_total else 0.0
    bottom_share = (bottom[metric] / grand_total * 100) if grand_total else 0.0
    findings.append(
        Finding(
            kind="concentration",
            title=f"'{top[dimension]}' leads with {top_share:.1f}% of the total",
            detail=(
                f"{top[dimension]} contributes {fmt_number(top[metric])} of "
                f"{fmt_number(grand_total)} across {len(working)} {dimension} values. "
                f"Lowest is {bottom[dimension]} at {fmt_number(bottom[metric])} "
                f"({bottom_share:.1f}% of total)."
            ),
            severity="info",
            value=float(top_share),
            evidence={
                "top": str(top[dimension]),
                "top_value": float(top[metric]),
                "top_share_pct": float(top_share),
                "bottom": str(bottom[dimension]),
                "bottom_value": float(bottom[metric]),
                "bottom_share_pct": float(bottom_share),
                "total": float(grand_total),
                "groups": len(working),
            },
        )
    )

    # Pareto: how few values produce 80% of the total.
    if len(working) >= 4 and grand_total:
        cumulative = working[metric].cumsum() / grand_total * 100
        count_80 = int((cumulative < 80).sum() + 1)
        if count_80 <= max(2, len(working) // 3):
            findings.append(
                Finding(
                    kind="pareto",
                    title=f"{count_80} of {len(working)} {dimension} values drive 80% of the total",
                    detail=(
                        "Outcome is concentrated rather than evenly spread; "
                        "effort spent on the long tail has limited leverage."
                    ),
                    severity="info",
                    value=float(count_80),
                    evidence={"count_80pct": count_80, "groups": len(working)},
                )
            )

    # Spread: is the leader an outlier relative to the rest?
    if len(working) >= 3:
        others = working.iloc[1:][metric]
        if others.max() > 0 and top[metric] / others.max() >= 2:
            findings.append(
                Finding(
                    kind="outlier",
                    title=f"'{top[dimension]}' is more than 2x the runner-up",
                    detail=(
                        f"{fmt_number(top[metric])} vs {fmt_number(others.max())} -- "
                        "a single-value dependency worth de-risking."
                    ),
                    severity="warning",
                    value=float(top[metric] / others.max()),
                )
            )

    return working, findings


def _quality(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    nulls = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "null_cells": nulls,
        "null_pct": round(nulls / total_cells * 100, 2) if total_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def analyze(df: pd.DataFrame, question: str = "", layer=None) -> Analysis:
    if df is None:
        df = pd.DataFrame()

    metric_cols, time_col, dimension_col = _pick_columns(df)
    kpis = _kpis(df, metric_cols, layer=layer)
    findings: list[Finding] = []
    timeseries = None
    breakdown = None

    if df.empty:
        findings.append(
            Finding(
                kind="empty",
                title="No rows returned",
                detail="The query ran successfully but matched no data. Widen the filters or check the date range.",
                severity="warning",
            )
        )
        return Analysis(
            rows=0,
            columns=list(df.columns),
            kpis=kpis,
            findings=findings,
            metric_cols=metric_cols,
            time_col=time_col,
            dimension_col=dimension_col,
            quality=_quality(df),
        )

    primary_metric = metric_cols[0] if metric_cols else None

    if primary_metric and time_col and df[time_col].nunique(dropna=True) >= 3:
        timeseries, trend_findings = _trend_findings(df, time_col, primary_metric)
        findings.extend(trend_findings)

    if primary_metric and dimension_col and df[dimension_col].nunique(dropna=True) > 1:
        breakdown, breakdown_findings = _breakdown_findings(df, dimension_col, primary_metric, 0.0)
        findings.extend(breakdown_findings)

    if df.shape[0] == 1:
        findings.append(
            Finding(
                kind="single_row",
                title="Single-row result",
                detail="Only one record matched, so trends and comparisons are not meaningful here.",
                severity="info",
            )
        )

    if df.shape[0] > 1 and primary_metric:
        series = pd.to_numeric(df[primary_metric], errors="coerce")
        negatives = int((series < 0).sum())
        if negatives:
            findings.append(
                Finding(
                    kind="data_quality",
                    title=f"{negatives} negative value(s) in {primary_metric}",
                    detail="Negative amounts can indicate refunds or a join producing duplicated rows. Verify before quoting totals.",
                    severity="warning",
                    value=float(negatives),
                )
            )

    return Analysis(
        rows=int(df.shape[0]),
        columns=list(df.columns),
        kpis=kpis,
        findings=findings,
        metric_cols=metric_cols,
        time_col=time_col,
        dimension_col=dimension_col,
        timeseries=timeseries,
        breakdown=breakdown,
        quality=_quality(df),
    )
