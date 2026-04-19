"""Markdown report generator for China value-investment reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from valueinvestor.data.models import CompanyAnalysis, InvestmentReport


def _fmt_number(value: float | None, decimals: int = 1, suffix: str = "") -> str:
    """Return a human-readable string for a number, or '—' when *None*."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def _fmt_billions(value: float | None) -> str:
    """Format a raw RMB amount as billions (¥X.X B)."""
    if value is None:
        return "—"
    return f"¥{value / 1e9:,.1f} B"


def _fmt_pct(value: float | None) -> str:
    """Format a ratio (0.15) or already-percentage value as 'XX.X%'."""
    if value is None:
        return "—"
    display = value * 100.0 if abs(value) < 1.0 else value
    return f"{display:.1f}%"


class MarkdownReportGenerator:
    """Produces a full Markdown document from an :class:`InvestmentReport`.

    This generator supports simple post-processing translations when the
    report's config_summary requests a different output language (e.g. zh-CN).
    """

    def generate(self, report: InvestmentReport) -> str:
        """Return the complete Markdown string for *report*."""
        # Determine output language (defaults to English)
        cfg = report.config_summary or {}
        self.output_language = cfg.get("output_language", "en")

        parts: list[str] = [
            self._heading(report),
            self._executive_summary(report),
            self._portfolio_overview(report),
            self._detailed_analyses(report),
            self._disclaimer(),
        ]
        return "\n".join(parts)

    def save(self, report: InvestmentReport, output_dir: str = "reports") -> str:
        """Generate the report and write it to *output_dir*.

        Returns the absolute path to the saved file.
        """
        md_content = self.generate(report)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = out / f"{date_str}_china_value_report.md"
        filepath.write_text(md_content, encoding="utf-8")
        return str(filepath)

    # ------------------------------------------------------------------
    # Internal section builders
    # ------------------------------------------------------------------

    def _heading(self, report: InvestmentReport) -> str:
        date_str = report.generated_at[:10] if report.generated_at else "N/A"
        if self.output_language == "zh-CN":
            return (
                f"# {report.title}\n"
                f"*生成时间: {date_str}*\n"
            )
        return (
            f"# {report.title}\n"
            f"*Generated: {date_str}*\n"
        )

    def _executive_summary(self, report: InvestmentReport) -> str:
        cfg = report.config_summary or {}
        amount = cfg.get("amount_rmb", 500_000)
        ret_min = cfg.get("target_return_min", 0.20)
        ret_max = cfg.get("target_return_max", 2.0)
        cycle_min = cfg.get("cycle_months_min", 6)
        cycle_max = cfg.get("cycle_months_max", 24)

        if self.output_language == "zh-CN":
            return (
                "## 执行摘要\n"
                f"- 筛选公司总数: {report.total_screened}\n"
                f"- 入选候选数: {report.total_candidates}\n"
                f"- 投资金额: ¥{amount:,} RMB\n"
                f"- 目标回报: {int(float(ret_min) * 100)}%–{int(float(ret_max) * 100)}%\n"
                f"- 投资期限: {cycle_min}–{cycle_max} 个月\n"
            )

        return (
            "## Executive Summary\n"
            f"- Total companies screened: {report.total_screened}\n"
            f"- Candidates selected: {report.total_candidates}\n"
            f"- Investment amount: ¥{amount:,} RMB\n"
            f"- Target return: {int(float(ret_min) * 100)}%–{int(float(ret_max) * 100)}%\n"
            f"- Investment horizon: {cycle_min}–{cycle_max} months\n"
        )

    def _portfolio_overview(self, report: InvestmentReport) -> str:
        if self.output_language == "zh-CN":
            lines = [
                "## 投资组合概览\n",
                "| 排名 | 代码 | 公司 | 行业 | 评分 | 权重 |",
                "|------|------|------|------|------|------|",
            ]
            for ca in report.candidates:
                sr = ca.screening_result
                weight = f"{ca.portfolio_weight:.0f}%" if ca.portfolio_weight is not None else "—"
                lines.append(
                    f"| {sr.rank} "
                    f"| {ca.ticker} "
                    f"| {ca.company_name} "
                    f"| {sr.company.sector or '—'} "
                    f"| {sr.composite_score:.1f} "
                    f"| {weight} |"
                )
            lines.append("")
            return "\n".join(lines)

        lines = [
            "## Portfolio Overview\n",
            "| Rank | Ticker | Company | Sector | Score | Weight |",
            "|------|--------|---------|--------|-------|--------|",
        ]
        for ca in report.candidates:
            sr = ca.screening_result
            weight = f"{ca.portfolio_weight:.0f}%" if ca.portfolio_weight is not None else "—"
            lines.append(
                f"| {sr.rank} "
                f"| {ca.ticker} "
                f"| {ca.company_name} "
                f"| {sr.company.sector or '—'} "
                f"| {sr.composite_score:.1f} "
                f"| {weight} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _detailed_analyses(self, report: InvestmentReport) -> str:
        if self.output_language == "zh-CN":
            sections = ["## 详细分析\n"]
        else:
            sections = ["## Detailed Analysis\n"]
        for idx, ca in enumerate(report.candidates, start=1):
            sections.append(self._company_section(idx, ca))
        return "\n".join(sections)

    def _company_section(self, idx: int, ca: CompanyAnalysis) -> str:
        sr = ca.screening_result
        weight = f"{ca.portfolio_weight:.0f}%" if ca.portfolio_weight is not None else "—"

        if self.output_language == "zh-CN":
            parts: list[str] = [
                f"### {idx}. {ca.company_name} ({ca.ticker})",
                f"**综合评分: {sr.composite_score:.1f} | 建议权重: {weight}**\n",
                self._financials_table(sr),
            ]
        else:
            parts: list[str] = [
                f"### {idx}. {ca.company_name} ({ca.ticker})",
                f"**Composite Score: {sr.composite_score:.1f} | Recommended Weight: {weight}**\n",
                self._financials_table(sr),
            ]

        # LLM analysis dimensions
        for dim in ca.analyses:
            # The LLM is instructed to produce JSON in Simplified Chinese when requested,
            # so we just render titles/content as-is here.
            parts.append(f"#### {dim.title}\n{dim.content}\n")

        # Key milestones
        if ca.key_milestones:
            if self.output_language == "zh-CN":
                parts.append("**需要关注的关键里程碑：**")
            else:
                parts.append("**Key Milestones to Watch:**")
            for ms in ca.key_milestones:
                parts.append(f"- {ms}")
            parts.append("")

        parts.append("---\n")
        return "\n".join(parts)

    def _financials_table(self, sr) -> str:
        """Build a two-column key-financials table with optional translation."""
        v = sr.valuation
        f = sr.financials
        rows = [
            ("Market Cap", _fmt_billions(v.market_cap_rmb or sr.company.market_cap_rmb)),
            ("Price", _fmt_number(v.price, decimals=2)),
            ("PE Ratio (TTM)", _fmt_number(v.pe_ratio)),
            ("PE Ratio (Fwd)", _fmt_number(v.pe_forward)),
            ("PB Ratio", _fmt_number(v.pb_ratio)),
            ("PS Ratio", _fmt_number(v.ps_ratio)),
            ("PEG Ratio", _fmt_number(v.peg_ratio)),
            ("Dividend Yield", _fmt_pct(v.dividend_yield)),
            ("EV/EBITDA", _fmt_number(v.ev_to_ebitda)),
            ("ROE", _fmt_pct(f.roe)),
            ("ROA", _fmt_pct(f.roa)),
            ("Gross Margin", _fmt_pct(f.gross_margin)),
            ("Net Margin", _fmt_pct(f.net_margin)),
            ("Debt / Equity", _fmt_number(f.debt_to_equity, decimals=2)),
            ("Current Ratio", _fmt_number(f.current_ratio, decimals=2)),
        ]

        if self.output_language == "zh-CN":
            label_map = {
                "Market Cap": "市值",
                "Price": "股价",
                "PE Ratio (TTM)": "市盈率（TTM）",
                "PE Ratio (Fwd)": "市盈率（预测）",
                "PB Ratio": "市净率",
                "PS Ratio": "市销率",
                "PEG Ratio": "PEG",
                "Dividend Yield": "股息率",
                "EV/EBITDA": "EV/EBITDA",
                "ROE": "净资产收益率（ROE）",
                "ROA": "总资产收益率（ROA）",
                "Gross Margin": "毛利率",
                "Net Margin": "净利率",
                "Debt / Equity": "负债/股东权益比",
                "Current Ratio": "流动比率",
            }
            lines = [
                "#### 主要财务指标",
                "| 指标 | 数值 |",
                "|------|------|",
            ]
            for label, val in rows:
                lbl = label_map.get(label, label)
                lines.append(f"| {lbl} | {val} |")
            lines.append("")
            return "\n".join(lines)

        lines = [
            "#### Key Financials",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for label, val in rows:
            lines.append(f"| {label} | {val} |")
        lines.append("")
        return "\n".join(lines)

    def _disclaimer(self) -> str:
        if self.output_language == "zh-CN":
            return (
                "## 免责声明\n"
                "本报告由自动化分析工具生成，不构成专业金融建议。所示信息来源于公开数据和AI驱动的分析，可能存在不准确之处。投资者在做出投资决策前应进行独立尽职调查并咨询专业投资顾问。历史业绩并不代表未来表现。\n"
            )

        return (
            "## Disclaimer\n"
            "This report is generated by an automated analysis tool and does not "
            "constitute professional financial advice. The information presented is "
            "derived from publicly available data and AI-driven analysis, which may "
            "contain inaccuracies. Investors should conduct their own due diligence "
            "and consult qualified financial advisors before making any investment "
            "decisions. Past performance is not indicative of future results.\n"
        )
