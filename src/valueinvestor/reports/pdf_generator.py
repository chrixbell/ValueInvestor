"""PDF report generator – converts Markdown reports to styled PDFs.

Uses the ``weasyprint`` CLI (install via ``brew install weasyprint`` on macOS)
to avoid Python library / system dependency conflicts.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
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

PDF_CSS = """\
@page {
    size: A4;
    margin: 2cm 1.8cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9px;
        color: #888;
    }
}

body {
    font-family: "Helvetica Neue", Helvetica, Arial, "PingFang SC",
                 "Microsoft YaHei", sans-serif;
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

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
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

        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "toc"],
        )
        full_html = _HTML_TEMPLATE.format(
            title="China Value Investment Report",
            css=PDF_CSS,
            body=html_body,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(full_html)
            html_path = tmp.name

        try:
            subprocess.run(
                [_WEASYPRINT_BIN, html_path, str(out)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "weasyprint failed:\n%s" % exc.stderr.strip()
            ) from exc
        finally:
            Path(html_path).unlink(missing_ok=True)

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
        pdf_path = out / f"{date_str}_china_value_multi_timeframe.pdf"
        return self.generate(md_content, str(pdf_path))
