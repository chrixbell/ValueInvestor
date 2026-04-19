"""PDF report generator – converts Markdown reports to styled PDFs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import markdown

from valueinvestor.data.models import InvestmentReport
from valueinvestor.reports.md_generator import MarkdownReportGenerator

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
<html lang="en">
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


class PDFReportGenerator:
    """Converts Markdown content (or an :class:`InvestmentReport`) to a styled PDF."""

    def generate(self, md_content: str, output_path: str) -> str:
        """Convert *md_content* (Markdown string) to a PDF file at *output_path*.

        Returns the absolute path of the generated PDF.
        """
        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "toc"],
        )
        full_html = _HTML_TEMPLATE.format(
            title="China Value Investment Report",
            css=PDF_CSS,
            body=html_body,
        )

        from weasyprint import HTML  # lazy import – requires system libs

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=full_html).write_pdf(str(out))
        return str(out)

    def generate_from_report(
        self,
        report: InvestmentReport,
        output_dir: str = "reports",
    ) -> str:
        """Generate a PDF directly from an :class:`InvestmentReport`.

        Uses :class:`MarkdownReportGenerator` internally to produce the
        intermediate Markdown, then converts it to PDF.

        Returns the absolute path of the generated PDF.
        """
        md_gen = MarkdownReportGenerator()
        md_content = md_gen.generate(report)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        pdf_path = out / f"{date_str}_china_value_report.pdf"
        return self.generate(md_content, str(pdf_path))
