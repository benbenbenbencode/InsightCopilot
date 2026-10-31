"""Chart specification and rendering.

One place decides *which* chart suits a dataframe; two renderers consume that
decision:

  * `render_png`  -> matplotlib, bytes for PDF/PPTX embedding
  * `render_altair` -> Altair, for the interactive Streamlit dashboard

Keeping the spec separate means the exported PDF and the on-screen dashboard
always show the same chart, not two hand-written variants.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Muted, print-friendly palette that matches the dashboard accent colour.
PALETTE = ["#6c5ce7", "#00b894", "#0984e3", "#fdcb6e", "#e17055", "#00cec9"]


@dataclass
class ChartSpec:
    kind: str  # "line" | "bar" | "hist" | "scatter" | "none"
    x: str | None
    y: str | None
    title: str
    subtitle: str = ""
    top_n: int = 12

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "title": self.title,
            "subtitle": self.subtitle,
        }


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def pick_chart(df: pd.DataFrame, time_col: str | None = None, dimension_col: str | None = None) -> ChartSpec:
    if df is None or df.empty:
        return ChartSpec(kind="none", x=None, y=None, title="No data")

    numerics = _numeric_columns(df)
    if not numerics:
        return ChartSpec(kind="none", x=None, y=None, title="No numeric column to plot")

    metric = numerics[0]

    if time_col and time_col in df.columns and df[time_col].nunique() >= 3:
        return ChartSpec(
            kind="line",
            x=time_col,
            y=metric,
            title=f"{metric.replace('_', ' ').title()} over time",
        )

    if dimension_col and dimension_col in df.columns and df[dimension_col].nunique() > 1:
        return ChartSpec(
            kind="bar",
            x=dimension_col,
            y=metric,
            title=f"{metric.replace('_', ' ').title()} by {dimension_col.replace('_', ' ')}",
        )

    if len(numerics) >= 2:
        return ChartSpec(
            kind="scatter",
            x=numerics[0],
            y=numerics[1],
            title=f"{numerics[1].replace('_', ' ').title()} vs {numerics[0].replace('_', ' ')}",
        )

    return ChartSpec(
        kind="hist",
        x=metric,
        y=None,
        title=f"Distribution of {metric.replace('_', ' ')}",
    )


def _prepare(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    working = df.copy()

    if spec.kind == "bar":
        working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
        working = working.dropna(subset=[spec.y])
        working = working.nlargest(spec.top_n, spec.y)

    if spec.kind == "line":
        working[spec.x] = pd.to_datetime(working[spec.x], errors="coerce")
        working[spec.y] = pd.to_numeric(working[spec.y], errors="coerce")
        working = working.dropna(subset=[spec.x, spec.y]).sort_values(spec.x)

    return working


def render_png(df: pd.DataFrame, spec: ChartSpec | None = None, width: float = 9.0, height: float = 4.2, dpi: int = 170, show_title: bool = True) -> bytes | None:
    """Render to PNG bytes. Returns None when there is nothing to draw."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    spec = spec or pick_chart(df)
    if spec.kind == "none":
        return None

    working = _prepare(df, spec)
    if working.empty:
        return None

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)

    def compact(value, _pos=None):
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.0f}K"
        return f"{value:.0f}"

    if spec.kind == "line":
        ax.plot(working[spec.x], working[spec.y], color=PALETTE[0], linewidth=2.2, marker="o", markersize=3.5)
        ax.fill_between(working[spec.x], working[spec.y], color=PALETTE[0], alpha=0.12)
        fig.autofmt_xdate(rotation=30)

    elif spec.kind == "bar":
        labels = [str(v)[:22] for v in working[spec.x]]
        bars = ax.barh(labels[::-1], working[spec.y][::-1], color=PALETTE[0], height=0.62)
        for bar in bars:
            width_value = bar.get_width()
            ax.text(
                width_value,
                bar.get_y() + bar.get_height() / 2,
                f" {compact(width_value)}",
                va="center",
                fontsize=8.5,
                color="#444444",
            )

    elif spec.kind == "scatter":
        ax.scatter(working[spec.x], working[spec.y], color=PALETTE[0], alpha=0.65, s=26)

    else:  # hist
        values = pd.to_numeric(working[spec.x], errors="coerce").dropna()
        ax.hist(values, bins=24, color=PALETTE[0], alpha=0.85)

    # Only the *numeric* axis gets the compact formatter. Applying it to a
    # categorical axis (barh) overwrites the category names with tick indices.
    if spec.kind == "bar":
        ax.xaxis.set_major_formatter(FuncFormatter(compact))
        ax.xaxis.get_offset_text().set_visible(False)
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(compact))

    if spec.kind in ("line", "bar"):
        ax.set_xlabel("")
        ax.set_ylabel("")
    if show_title:
        ax.set_title(spec.title, loc="left", fontsize=12, color="#222222", pad=12)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return buffer.getvalue()


def render_altair(df: pd.DataFrame, spec: ChartSpec | None = None):
    """Altair chart for Streamlit. Returns None when there is nothing to draw."""
    import altair as alt

    spec = spec or pick_chart(df)
    if spec.kind == "none":
        return None

    working = _prepare(df, spec)
    if working.empty:
        return None

    if spec.kind == "line":
        chart = (
            alt.Chart(working)
            .mark_line(point=True, strokeWidth=2.4, color=PALETTE[0])
            .encode(
                x=alt.X(f"{spec.x}:T", title=None),
                y=alt.Y(f"{spec.y}:Q", title=None),
                tooltip=[spec.x, spec.y],
            )
        )
    elif spec.kind == "bar":
        chart = (
            alt.Chart(working)
            .mark_bar(color=PALETTE[0], cornerRadiusEnd=3)
            .encode(
                x=alt.X(f"{spec.y}:Q", title=None),
                y=alt.Y(f"{spec.x}:N", sort="-x", title=None),
                tooltip=[spec.x, spec.y],
            )
        )
    elif spec.kind == "scatter":
        chart = (
            alt.Chart(working)
            .mark_circle(size=55, color=PALETTE[2], opacity=0.7)
            .encode(
                x=alt.X(f"{spec.x}:Q", title=spec.x),
                y=alt.Y(f"{spec.y}:Q", title=spec.y),
                tooltip=[spec.x, spec.y],
            )
        )
    else:
        chart = (
            alt.Chart(working)
            .mark_bar(color=PALETTE[0])
            .encode(
                x=alt.X(f"{spec.x}:Q", bin=alt.Bin(maxbins=24), title=spec.x),
                y=alt.Y("count():Q", title="count"),
            )
        )

    return chart.properties(height=320, title=spec.title).configure_view(stroke=None)
