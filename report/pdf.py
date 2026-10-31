"""PDF report rendering with fpdf2.

Layout: cover band -> executive summary -> KPI cards -> chart -> findings ->
recommendations -> caveats -> data table -> provenance. Everything is drawn
from the ReportBundle, so the PDF can never drift from what the dashboard
showed.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from fpdf import FPDF

from core.analysis import fmt_number
from report.model import ReportBundle

# fpdf2 logs a subsetting notice for font tables it does not understand
# (e.g. "MERG NOT subset"). It is harmless and clutters CLI output.
warnings.filterwarnings("ignore", message=".*NOT subset.*")

INK = (24, 24, 32)
MUTED = (110, 110, 122)
ACCENT = (108, 92, 231)
ACCENT_SOFT = (238, 235, 255)
BORDER = (226, 226, 232)
PANEL = (248, 248, 252)
POSITIVE = (0, 150, 120)
WARNING = (196, 120, 20)
CRITICAL = (190, 50, 50)

SEVERITY_COLORS = {
    "positive": POSITIVE,
    "warning": WARNING,
    "critical": CRITICAL,
    "info": ACCENT,
}

FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def _png_size(data: bytes) -> tuple[int, int] | None:
    if not data or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def _ascii(text: str) -> str:
    return (
        str(text)
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2022", "-")
        .replace("\u00a0", " ")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


class ReportPDF(FPDF):
    def __init__(self, bundle: ReportBundle):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.bundle = bundle
        self.unicode_ok = False
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=20)
        self._register_fonts()

    def _register_fonts(self):
        for regular, bold in FONT_CANDIDATES:
            if os.path.exists(regular) and os.path.exists(bold):
                try:
                    self.add_font("Body", "", regular)
                    self.add_font("Body", "B", bold)
                    self.unicode_ok = True
                    return
                except Exception:
                    continue
        self.unicode_ok = False

    def text_of(self, value: str) -> str:
        return str(value) if self.unicode_ok else _ascii(value)

    def font(self, size: float, weight: str = ""):
        self.set_font("Body" if self.unicode_ok else "Helvetica", weight, size)

    def wrap(self, width: float, line_height: float, text: str):
        """Left-aligned multi-line text.

        fpdf2's multi_cell justifies by default, which stretches large bold
        headlines into something unreadable. Always pass align="L".
        """
        self.multi_cell(width, line_height, self.text_of(text), align="L", new_x="LMARGIN", new_y="NEXT")

    # -- chrome ------------------------------------------------------------

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.font(7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.text_of("Retail Insight Copilot - generated locally"), align="L")
        self.set_x(-30)
        self.cell(14, 5, f"{self.page_no()}/{self.pages_count}", align="R")

    # -- building blocks ---------------------------------------------------

    def section_title(self, text: str, number: str = ""):
        # Avoid stranding a heading at the bottom of a page with nothing under it.
        if self.get_y() > 250:
            self.add_page()

        self.ln(3)
        self.font(8)
        self.set_text_color(*ACCENT)
        label = f"{number}  {text.upper()}" if number else text.upper()
        self.cell(0, 5, self.text_of(label), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        y = self.get_y() + 0.5
        self.line(16, y, 194, y)
        self.ln(4)

    def kpi_cards(self, kpis):
        if not kpis:
            return
        cards = kpis[:4]
        gap = 4
        total_width = 194 - 16
        card_w = (total_width - gap * (len(cards) - 1)) / len(cards)
        card_h = 20
        x = 16
        y = self.get_y()

        for kpi in cards:
            self.set_fill_color(*PANEL)
            self.set_draw_color(*BORDER)
            self.rect(x, y, card_w, card_h, style="DF")
            self.set_xy(x + 3, y + 2.5)
            self.font(6.8)
            self.set_text_color(*MUTED)
            self.cell(card_w - 6, 4, self.text_of(kpi.label.upper()[:30]))
            self.set_xy(x + 3, y + 7.5)
            self.font(13, "B")
            self.set_text_color(*INK)
            self.cell(card_w - 6, 8, self.text_of(kpi.display))
            x += card_w + gap

        self.set_y(y + card_h + 5)

    def paragraph(self, text: str, size: float = 9.8, color=INK):
        self.font(size)
        self.set_text_color(*color)
        self.wrap(0, 5.2, text)
        self.ln(1.5)

    def table(self, df, max_rows: int = 18):
        if df is None or df.empty:
            return

        columns = list(df.columns)[:7]
        rows = df[columns].head(max_rows)

        # Column widths weighted by the widest content, with a floor so numbers
        # stay readable and a cap so one long string cannot eat the page.
        usable = 178
        weights = []
        for column in columns:
            lengths = [len(str(column))]
            for value in rows[column].tolist():
                lengths.append(len(fmt_number(value)) if isinstance(value, (int, float)) else len(str(value)))
            weights.append(min(max(max(lengths), 6), 34))

        scale = usable / sum(weights)
        widths = [w * scale for w in weights]

        self.font(7.8, "B")
        self.set_fill_color(*ACCENT_SOFT)
        self.set_text_color(*INK)
        self.set_draw_color(*BORDER)
        for column, width in zip(columns, widths):
            self.cell(width, 7, self.text_of(str(column)[:32]), border=0, fill=True)
        self.ln(7)

        self.font(7.8)
        for index, (_, row) in enumerate(rows.iterrows()):
            if self.get_y() > 262:
                break
            self.set_fill_color(*(PANEL if index % 2 == 0 else (255, 255, 255)))
            for column, width in zip(columns, widths):
                value = row[column]
                text = fmt_number(value) if isinstance(value, (int, float)) else str(value)
                if len(text) > 34:
                    text = text[:31] + "..."
                self.cell(width, 6.4, self.text_of(text), border=0, fill=True)
            self.ln(6.4)

        if len(df) > len(rows):
            self.ln(1)
            self.font(7.5)
            self.set_text_color(*MUTED)
            self.cell(0, 5, self.text_of(f"Showing {len(rows)} of {len(df)} rows."), new_x="LMARGIN", new_y="NEXT")


def build_pdf(bundle: ReportBundle, path: str | Path) -> Path:
    pdf = ReportPDF(bundle)
    pdf.add_page()

    # ---- cover band ------------------------------------------------------
    pdf.set_fill_color(*INK)
    pdf.rect(0, 0, 210, 44, style="F")

    pdf.set_xy(16, 12)
    pdf.set_text_color(255, 255, 255)
    pdf.font(19, "B")
    pdf.cell(0, 9, pdf.text_of(bundle.title), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(16)
    pdf.font(9.5)
    pdf.set_text_color(200, 200, 210)
    pdf.cell(0, 5, pdf.text_of(f"Question: {bundle.question}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(16)
    pdf.font(8)
    pdf.set_text_color(150, 150, 165)
    pdf.cell(
        0,
        5,
        pdf.text_of(f"Generated {bundle.stamp}  |  local model: {bundle.narrative.model or 'n/a'}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_y(50)
    pdf.set_text_color(*INK)

    # ---- executive summary ----------------------------------------------
    pdf.section_title("Executive summary", "1")
    if bundle.narrative.headline:
        pdf.font(13, "B")
        pdf.set_text_color(*INK)
        pdf.wrap(0, 6.5, bundle.narrative.headline)
        pdf.ln(1.5)

    pdf.paragraph(bundle.narrative.summary)

    if bundle.narrative.key_points:
        pdf.ln(1)
        for point in bundle.narrative.key_points:
            pdf.set_x(18)
            pdf.font(9.5)
            pdf.set_text_color(*ACCENT)
            pdf.cell(4, 5, "-")
            pdf.set_text_color(*INK)
            pdf.wrap(170, 5, point)
            pdf.ln(0.6)

    # ---- KPIs ------------------------------------------------------------
    pdf.section_title("Key metrics", "2")
    pdf.kpi_cards(bundle.analysis.kpis)

    # ---- chart -----------------------------------------------------------
    if bundle.chart_png:
        size = _png_size(bundle.chart_png)
        pdf.section_title("Visual", "3")
        width_mm = 178
        height_mm = width_mm * size[1] / size[0] if size else 82
        if pdf.get_y() + height_mm > 268:
            pdf.add_page()
        if height_mm > 120:
            height_mm = 120
            width_mm = height_mm * size[0] / size[1]
        pdf.image(bundle.chart_png, x=16, y=pdf.get_y(), w=width_mm, h=height_mm)
        pdf.set_y(pdf.get_y() + height_mm + 3)

        if bundle.chart_spec.get("title"):
            pdf.font(7.8)
            pdf.set_text_color(*MUTED)
            pdf.cell(
                0,
                4,
                pdf.text_of(f"{bundle.chart_spec['title']}  ·  {bundle.analysis.rows} rows"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(2)

    # ---- findings --------------------------------------------------------
    if bundle.analysis.findings:
        pdf.section_title("What the data shows", "4")
        for idx, finding in enumerate(bundle.analysis.findings, 1):
            if pdf.get_y() > 258:
                pdf.add_page()
            color = SEVERITY_COLORS.get(finding.severity, ACCENT)
            y = pdf.get_y()
            pdf.set_fill_color(*color)
            pdf.rect(16, y + 1.6, 1.6, 1.6, style="F")

            pdf.set_xy(20, y)
            pdf.font(9.6, "B")
            pdf.set_text_color(*INK)
            pdf.wrap(174, 5.2, f"{idx}. {finding.title}")

            pdf.set_x(20)
            pdf.font(9)
            pdf.set_text_color(*MUTED)
            pdf.wrap(174, 4.9, finding.detail)
            pdf.ln(1.8)

    # ---- recommendation & risks -----------------------------------------
    if bundle.narrative.recommendations:
        pdf.section_title("Recommended actions", "5")
        for idx, item in enumerate(bundle.narrative.recommendations, 1):
            pdf.set_x(18)
            pdf.font(9.5, "B")
            pdf.set_text_color(*ACCENT)
            pdf.cell(5, 5, f"{idx}.")
            pdf.set_text_color(*INK)
            pdf.wrap(169, 5, item)
            pdf.ln(0.8)

    if bundle.narrative.risks:
        pdf.section_title("Caveats", "6")
        for item in bundle.narrative.risks:
            pdf.set_x(18)
            pdf.font(8.8)
            pdf.set_text_color(*WARNING)
            pdf.cell(4, 4.8, "!")
            pdf.set_text_color(*MUTED)
            pdf.wrap(168, 4.8, item)
            pdf.ln(0.5)

    # ---- data & provenance ----------------------------------------------
    pdf.add_page()
    pdf.section_title("Result data", "7")
    pdf.table(bundle.df)
    pdf.ln(4)

    pdf.section_title("Provenance", "8")
    pdf.font(8.2)
    pdf.set_text_color(*MUTED)
    pdf.paragraph(f"Query over {bundle.analysis.rows} returned row(s):", 8.2, MUTED)

    pdf.set_font("Courier", "", 7.4)
    pdf.set_text_color(*INK)
    pdf.wrap(0, 4.2, _ascii(bundle.sql))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return output
