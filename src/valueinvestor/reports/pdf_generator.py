"""PDF report generator – converts Markdown reports to styled PDFs.

Uses the ``weasyprint`` CLI (install via ``brew install weasyprint`` on macOS)
to avoid Python library / system dependency conflicts.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import markdown

from valueinvestor.data.models import InvestmentReport, MultiTimeframeReport
from valueinvestor.reports.md_generator import (
    MarkdownReportGenerator,
    MultiTimeframeMarkdownReportGenerator,
)

# ---------------------------------------------------------------------------
# Professional finance-themed CSS
# ---------------------------------------------------------------------------

_EMBEDDED_GB18030_FAMILY = "ValueInvestorGB18030Heiti"
_GB18030_FONT_FACES = (
    (300, "STHeiti:style=Light"),
    (400, "STHeiti:style=Regular"),
    (700, "STHeiti:style=Regular"),
)

PDF_CSS = """\
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: "ValueInvestorGB18030Heiti", "STHeiti", "Heiti SC",
                     "Hiragino Sans GB", "Songti SC", serif;
        font-size: 9px;
        color: #888;
    }
}

body {
    font-family: "ValueInvestorGB18030Heiti", "STHeiti", "Heiti SC",
                 "Hiragino Sans GB", "Songti SC", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
}

h1 {
    color: #1a3c6e;
    border-bottom: 3px solid #1a3c6e;
    padding-bottom: 8px;
    font-size: 22pt;
}

h2 {
    color: #1a3c6e;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 28px;
    font-size: 16pt;
}

h3 {
    color: #2a5a9e;
    margin-top: 22px;
    font-size: 13pt;
}

h4 {
    color: #444;
    margin-top: 14px;
    font-size: 11pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 18px 0;
    font-size: 10pt;
}

th {
    background-color: #1a3c6e;
    color: #fff;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 5px 10px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) td {
    background-color: #f7f9fc;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 30px 0;
}

em {
    color: #555;
    font-style: normal;
}

strong {
    color: #1a3c6e;
}

ul {
    margin: 6px 0;
    padding-left: 22px;
}

li {
    margin-bottom: 3px;
}

p {
    margin: 8px 0;
}
"""


@lru_cache(maxsize=8)
def _fontconfig_font_path(query: str) -> Path | None:
    """Return a concrete font file path resolved by fontconfig."""
    try:
        proc = subprocess.run(
            ["fc-match", "-f", "%{file}", query],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    font_path = Path(proc.stdout.strip())
    if not font_path.exists():
        return None
    return font_path


def _embedded_font_css(font_dir: Path | None = None) -> str:
    """Build @font-face rules for Apple Preview-friendly GB18030 Chinese fonts."""
    del font_dir
    font_rules = []
    for weight, query in _GB18030_FONT_FACES:
        font_path = _fontconfig_font_path(query)
        if font_path is None:
            continue

        font_rules.append(
            f"""\
@font-face {{
    font-family: "{_EMBEDDED_GB18030_FAMILY}";
    src: url("{font_path.as_uri()}");
    font-weight: {weight};
    font-style: normal;
}}
"""
        )

    if not font_rules:
        return ""

    return "\n".join(font_rules) + "\n"


def _pdf_css(font_dir: Path | None = None) -> str:
    """Return the complete PDF CSS with optional embedded local font rules."""
    return _embedded_font_css(font_dir) + PDF_CSS

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="gb18030">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Path to the weasyprint CLI (brew-installed)
# ---------------------------------------------------------------------------

_WEASYPRINT_BIN = "/opt/homebrew/bin/weasyprint"


def _weasyprint_available() -> bool:
    """Return True if the weasyprint CLI can be found."""
    return Path(_WEASYPRINT_BIN).exists()


class PDFReportGenerator:
    """Converts Markdown content (or a report object) to a styled PDF.

    Uses the ``weasyprint`` CLI under the hood, which must be installed
    separately (``brew install weasyprint`` on macOS).
    """

    def generate(self, md_content: str, output_path: str) -> str:
        """Convert *md_content* (Markdown string) to a PDF file at *output_path*.

        Returns the absolute path of the generated PDF.
        """
        if not _weasyprint_available():
            raise RuntimeError(
                "weasyprint CLI not found at %s. "
                "Install it via: brew install weasyprint"
                % _WEASYPRINT_BIN
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="valueinvestor_pdf_") as tmp:
            tmp_dir = Path(tmp)
            html_body = markdown.markdown(
                md_content,
                extensions=["tables", "fenced_code", "toc"],
            )
            full_html = _HTML_TEMPLATE.format(
                title="China Value Investment Report",
                css=_pdf_css(tmp_dir),
                body=html_body,
            )
            html_path = tmp_dir / "report.html"
            html_path.write_text(full_html, encoding="gb18030")

            try:
                subprocess.run(
                    [_WEASYPRINT_BIN, str(html_path), str(out)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "weasyprint failed:\n%s" % exc.stderr.strip()
                ) from exc

        return str(out)

    def generate_from_report(
        self,
        report: InvestmentReport,
        output_dir: str = "reports",
    ) -> str:
        """Generate a PDF directly from an :class:`InvestmentReport`.

        Returns the absolute path of the generated PDF.
        """
        md_gen = MarkdownReportGenerator()
        md_content = md_gen.generate(report)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        pdf_path = out / f"{date_str}_china_value_report.pdf"
        return self.generate(md_content, str(pdf_path))

    def generate_from_multi_timeframe_report(
        self,
        report: MultiTimeframeReport,
        output_dir: str = "reports",
    ) -> str:
        """Generate a PDF from a :class:`MultiTimeframeReport`.

        Returns the absolute path of the generated PDF.
        """
        md_gen = MultiTimeframeMarkdownReportGenerator()
        md_content = md_gen.generate(report)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        targets = list(report.results_by_horizon)
        if "1w" in targets and "6m" in targets:
            suffix = "dual_target"
        elif targets:
            suffix = f"{targets[0]}_target"
        else:
            suffix = "multi_timeframe"
        pdf_path = out / f"{date_str}_china_value_{suffix}.pdf"
        return self.generate(md_content, str(pdf_path))
