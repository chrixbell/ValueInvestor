"""Markdown report generator for China value-investment reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from valueinvestor.data.models import CompanyAnalysis, InvestmentReport, MultiTimeframeReport


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


# ---------------------------------------------------------------------------
# Multi-Timeframe Report Generator (always Chinese output)
# ---------------------------------------------------------------------------

_HORIZON_LABELS_ZH: dict = {
    "1m": "1个月",
    "3m": "3个月",
    "6m": "6个月",
}


class MultiTimeframeMarkdownReportGenerator:
    """Produce a Chinese-language Markdown report with ρ evaluation.

    Structure:
    1. 标题 + 执行摘要
    2. 评分算法解释（ρ 值与评分含义）
    3. 最佳投资标的（单一表格）
    4. 个股详细分析
    5. 免责声明
    """

    def generate(self, report: MultiTimeframeReport) -> str:
        """Return the full Markdown string."""
        results = report.results_by_horizon.get("6m", []) or next(iter(report.results_by_horizon.values()), [])
        sections = [
            self._heading(report),
            self._summary(report, results),
            self._scoring_explanation(report),
        ]
        sections.append(self._candidates_table(results))
        sections.append(self._detailed_analyses(report))
        sections.append(self._disclaimer())
        return "\n\n".join(sections)

    def save(self, report: MultiTimeframeReport, output_dir: str = "reports") -> str:
        """Generate and save the report; return the saved file path."""
        md_content = self.generate(report)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = out / f"{date_str}_china_value_multi_timeframe.md"
        filepath.write_text(md_content, encoding="utf-8")
        return str(filepath)

    def _heading(self, report: MultiTimeframeReport) -> str:
        date_str = report.generated_at[:10] if report.generated_at else "N/A"
        return f"# {report.title}\n*生成时间: {date_str}*"

    def _summary(self, report: MultiTimeframeReport, results: list) -> str:
        total_screened = report.total_screened
        n_candidates = len(results)
        lines = [
            "## 执行摘要\n",
            f"- 筛选公司总数: {total_screened}",
            f"- 入选候选数: {n_candidates} 只",
            f"- 个股分析覆盖: {len(report.company_analyses)} 只",
        ]
        # ρ summary line
        rhos = report.spearman_rhos
        if rhos:
            parts = []
            for h in ("1m", "3m", "6m"):
                lbl = _HORIZON_LABELS_ZH.get(h, h)
                rho = rhos.get(h)
                if rho is not None:
                    parts.append(f"{lbl} ρ={rho:.4f}")
            if parts:
                lines.append(f"- 当前评分算法预测力: {' · '.join(parts)}")
        return "\n".join(lines)

    def _scoring_explanation(self, report: MultiTimeframeReport) -> str:
        """Explain ρ values, ρ interpretation, and score interpretation."""
        rhos = report.spearman_rhos

        # Build ρ status lines
        rho_lines = []
        for h in ("1m", "3m", "6m"):
            lbl = _HORIZON_LABELS_ZH.get(h, h)
            rho = rhos.get(h) if rhos else None
            if rho is not None:
                quality = _rho_quality(rho)
                rho_lines.append(f"| {lbl} | {rho:.4f} | {quality} |")
            else:
                rho_lines.append(f"| {lbl} | 暂无数据 | — |")

        rho_table = "\n".join(rho_lines)

        return f"""## 评分算法说明

### Spearman ρ（秩相关系数）

ρ 值衡量综合评分与未来实际回报的**排序一致性**。取值范围 -1 到 +1：

| ρ 范围 | 含义 |
|--------|------|
| **0.20 ~ 0.30** | 良好 — 评分具有显著预测力，超出大多数单因子策略 |
| **0.10 ~ 0.20** | 一般 — 有一定预测力，但仍需改进 |
| **0.05 ~ 0.10** | 较弱 — 仅略优于随机选择 |
| **0.00 ~ 0.05** | 极弱 — 几乎无预测力 |
| **< 0.00** | 反向 — 评分与回报负相关，需重新审视算法 |

> 在金融领域，单因子的 ρ 通常不超过 ±0.10。多因子模型能达到 0.15+
> 已属于优秀水平。ρ 为 0 表示评分与回报无任何单调关系。

当前算法在不同时间框架的表现：

| 预测周期 | Spearman ρ | 评级 |
|----------|-----------|------|
{rho_table}

### 综合评分（0–100）

综合评分由三个因子通过**加权几何平均**合成：

- **价值因子（50%）**：PE、PB、PS、股息率（含自由现金流覆盖率调整）、EV/EBITDA、盈利收益率、自由现金流收益率等 12+ 项子指标
- **质量因子（35%）**：ROE × 毛利率 /（负债率 + 1）/ PE，在横截面上百分位排名
- **成长因子（15%）**：PEG 比率、预测PE相对历史PE的改善幅度

几何平均的特性是：**任何一项因子得分极低都会严重拖累综合评分**，因此高分股票必须在价值、质量、成长三个维度上均表现良好。

| 评分范围 | 含义 |
|----------|------|
| **60+** | 优秀 — 各项因子均衡且突出，是筛选系统能产生的最佳候选 |
| **50–60** | 良好 — 多数因子表现不错，综合吸引力较强 |
| **40–50** | 中等 — 部分因子存在短板 |
| **< 40** | 偏差 — 至少一项核心指标严重拖分 |

> 实际筛选结果通常在 50–65 分区间。100 分仅为理论最大值，现实中不存在
> 所有因子同时完美的股票。"""

    def _candidates_table(self, results: list) -> str:
        lines = [
            "## 最佳投资标的\n",
            "| 排名 | 代码 | 公司 | 行业 | 综合评分 |",
            "|------|------|------|------|----------|",
        ]
        for sr in results:
            lines.append(
                f"| {sr.rank} "
                f"| {sr.company.ticker} "
                f"| {sr.company.name} "
                f"| {sr.company.sector or '—'} "
                f"| {sr.composite_score:.1f} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _detailed_analyses(self, report: MultiTimeframeReport) -> str:
        sections = ["## 个股详细分析\n"]
        for idx, ca in enumerate(report.company_analyses, start=1):
            sections.append(self._company_section(idx, ca))
        return "\n".join(sections)

    def _company_section(self, idx: int, ca: CompanyAnalysis) -> str:
        sr = ca.screening_result
        weight = f"{ca.portfolio_weight:.0f}%" if ca.portfolio_weight is not None else "—"

        parts = [
            f"### {idx}. {ca.company_name} ({ca.ticker})",
            f"**综合评分: {sr.composite_score:.1f} | 建议权重: {weight}**\n",
            self._financials_table(sr),
        ]

        for dim in ca.analyses:
            parts.append(f"#### {dim.title}\n{dim.content}\n")

        if ca.key_milestones:
            parts.append("**需要关注的关键里程碑：**")
            for ms in ca.key_milestones:
                parts.append(f"- {ms}")
            parts.append("")

        parts.append("---\n")
        return "\n".join(parts)

    def _financials_table(self, sr) -> str:
        v = sr.valuation
        f = sr.financials
        rows = [
            ("市值", _fmt_billions(v.market_cap_rmb or sr.company.market_cap_rmb)),
            ("股价", _fmt_number(v.price, decimals=2)),
            ("市盈率（TTM）", _fmt_number(v.pe_ratio)),
            ("市盈率（预测）", _fmt_number(v.pe_forward)),
            ("市净率", _fmt_number(v.pb_ratio)),
            ("市销率", _fmt_number(v.ps_ratio)),
            ("PEG", _fmt_number(v.peg_ratio)),
            ("股息率", _fmt_pct(v.dividend_yield)),
            ("EV/EBITDA", _fmt_number(v.ev_to_ebitda)),
            ("净资产收益率（ROE）", _fmt_pct(f.roe)),
            ("总资产收益率（ROA）", _fmt_pct(f.roa)),
            ("毛利率", _fmt_pct(f.gross_margin)),
            ("净利率", _fmt_pct(f.net_margin)),
            ("负债/股东权益比", _fmt_number(f.debt_to_equity, decimals=2)),
            ("流动比率", _fmt_number(f.current_ratio, decimals=2)),
        ]
        lines = [
            "#### 主要财务指标",
            "| 指标 | 数值 |",
            "|------|------|",
        ]
        for label, val in rows:
            lines.append(f"| {label} | {val} |")
        lines.append("")
        return "\n".join(lines)

    def _disclaimer(self) -> str:
        return (
            "## 免责声明\n"
            "本报告由自动化分析工具生成，不构成专业金融建议。"
            "所示信息来源于公开数据和AI驱动的分析，可能存在不准确之处。"
            "投资者在做出投资决策前应进行独立尽职调查并咨询专业投资顾问。"
            "历史业绩并不代表未来表现。\n"
        )


def _rho_quality(rho: float) -> str:
    """Return a Chinese quality label for a Spearman ρ value."""
    if rho >= 0.20:
        return "优秀"
    elif rho >= 0.10:
        return "良好"
    elif rho >= 0.05:
        return "一般"
    elif rho >= 0.0:
        return "较弱"
    else:
        return "反向"

