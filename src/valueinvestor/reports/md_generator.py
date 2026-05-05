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


def _zh_financials_table(sr) -> str:
    """Render a Chinese key-financials Markdown table for *sr*."""
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


def _zh_disclaimer() -> str:
    """Return the standard Chinese investment disclaimer section."""
    return (
        "## 免责声明\n\n"
        "**本报告不是股票预测，不构成任何形式的投资建议。**\n\n"
        "这是一个基于过去10年（2016–2025）历史财务数据的算法驱动股票筛选系统。"
        "综合评分由价值、质量、成长、动量、协同、价值-成长六个因子通过量化模型计算得出，"
        "仅反映企业在这些维度上的相对排名，**不代表对未来股价的任何预测**。\n\n"
        "Spearman ρ 值衡量的是评分排序与历史实际回报之间的秩相关性，"
        "用于回测验证评分算法的有效性，不保证未来表现。\n\n"
        "所示信息来源于公开数据与AI驱动的分析，可能存在不准确或过时之处。"
        "投资者在做出任何投资决策前，应进行独立的尽职调查并咨询持牌专业投资顾问。"
        "历史业绩不代表未来表现，过往回测结果不构成对未来收益的承诺或预示。\n"
    )


def _rho_value_from_config(cfg: dict) -> float | None:
    """Return the current 6-month Spearman rho from a report config snapshot."""
    rhos = cfg.get("spearman_rhos")
    if not isinstance(rhos, dict):
        return None
    rho = rhos.get("6m")
    if rho is None:
        return None
    try:
        return float(rho)
    except (TypeError, ValueError):
        return None


def _zh_upfront_model_notes(cfg: dict) -> str:
    """Render the three upfront model, rho, and score explanation sections."""
    rho_6m = _rho_value_from_config(cfg)
    if rho_6m is None:
        rho_current = "当前报告未附带可用的 6 个月 Spearman ρ 数值。"
    else:
        rho_current = f"当前 6 个月 Spearman ρ = **{rho_6m:.4f}**，评级为 **{_rho_quality(rho_6m)}**。"

    return (
        "## 报告使用前说明\n\n"
        "### 1. 模型生成与免责声明\n\n"
        "本报告由模型自动生成，**不是财务预测，也不是股票推荐或买卖建议**。"
        "报告只基于过去约 10 年公司基本面数据、估值数据和量化评分系统，"
        "评估公司在未来 6 个月股票增长排序中的相对可能性。"
        "该结果只能作为研究线索，不应作为任何投资决策的唯一依据。\n\n"
        "### 2. 当前 6 个月 Spearman ρ 的含义\n\n"
        f"{rho_current} "
        "Spearman ρ 衡量的是模型综合评分排序与未来 6 个月实际收益排序之间的相关性，"
        "也就是“高分股票是否更倾向于在之后 6 个月取得更靠前的收益排名”。"
        "ρ 的取值范围是 **-1 到 +1**：+1 表示排序完全一致，0 表示没有单调排序关系，"
        "-1 表示排序完全相反。\n\n"
        "| ρ 范围 | 含义 |\n"
        "|--------|------|\n"
        "| **0.20 以上** | 优秀：在股票横截面排序中具有较强历史预测力 |\n"
        "| **0.10–0.20** | 良好：具备可观察的排序预测力 |\n"
        "| **0.05–0.10** | 一般：略优于随机排序 |\n"
        "| **0.00–0.05** | 较弱：预测力非常有限 |\n"
        "| **低于 0.00** | 反向：评分与未来收益排序负相关 |\n\n"
        "### 3. 综合评分的含义\n\n"
        "综合评分是 0–100 的相对评分，用来比较同一轮筛选中公司基本面、估值、质量、成长、动量和因子均衡性的综合表现。"
        "分数越高，表示该公司在模型历史回测框架下越接近未来 6 个月收益排序靠前的特征，"
        "但这**不保证**任何单一股票未来上涨。\n\n"
        "| 综合评分范围 | 含义 |\n"
        "|--------------|------|\n"
        "| **60+** | 优秀：各项因子较均衡，是模型能筛出的强候选区间 |\n"
        "| **50–60** | 良好：综合吸引力较强，但仍可能有短板 |\n"
        "| **40–50** | 中等：部分因子表现一般或不均衡 |\n"
        "| **低于 40** | 偏弱：至少一项核心因子明显拖累 |\n"
    )


def _en_upfront_model_notes(cfg: dict) -> str:
    """Render the upfront model, rho, and score explanation sections."""
    rho_6m = _rho_value_from_config(cfg)
    if rho_6m is None:
        rho_current = "No current 6-month Spearman rho value is attached to this report."
    else:
        rho_current = f"Current 6-month Spearman rho = **{rho_6m:.4f}**."

    return (
        "## Before Using This Report\n\n"
        "### 1. Model-Generated Disclaimer\n\n"
        "This report is generated by a model. It is **not** a financial forecast, stock pick, "
        "or buy/sell recommendation. It is based on roughly 10 years of company fundamental "
        "and valuation data and estimates relative likelihood of future 6-month stock-growth ranking.\n\n"
        "### 2. Current 6-Month Spearman Rho\n\n"
        f"{rho_current} Spearman rho measures rank correlation between model scores and actual "
        "future 6-month return rankings. Its range is **-1 to +1**: +1 is perfectly aligned, "
        "0 is no monotonic ranking relationship, and -1 is perfectly reversed.\n\n"
        "### 3. Score Value\n\n"
        "The composite score is a 0-100 relative score. Higher scores mean the company better "
        "matches characteristics that historically ranked better over the following 6 months, "
        "but this does not guarantee future performance.\n"
    )


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
            self._upfront_model_notes(report),
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

    def _upfront_model_notes(self, report: InvestmentReport) -> str:
        cfg = report.config_summary or {}
        if self.output_language == "zh-CN":
            return _zh_upfront_model_notes(cfg)
        return _en_upfront_model_notes(cfg)

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
                "| 排名 | 代码 | 公司 | 行业 | 综合评分 |",
                "|------|------|------|------|----------|",
            ]
            for ca in report.candidates:
                sr = ca.screening_result
                lines.append(
                    f"| {sr.rank} "
                    f"| {ca.ticker} "
                    f"| {ca.company_name} "
                    f"| {sr.company.sector or '—'} "
                    f"| {sr.composite_score:.1f} |"
                )
            lines.append("")
            return "\n".join(lines)

        lines = [
            "## Portfolio Overview\n",
            "| Rank | Ticker | Company | Sector | Score |",
            "|------|--------|---------|--------|-------|",
        ]
        for ca in report.candidates:
            sr = ca.screening_result
            lines.append(
                f"| {sr.rank} "
                f"| {ca.ticker} "
                f"| {ca.company_name} "
                f"| {sr.company.sector or '—'} "
                f"| {sr.composite_score:.1f} |"
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

        if self.output_language == "zh-CN":
            parts: list[str] = [
                f"### {idx}. {ca.company_name} ({ca.ticker})",
                f"**综合评分: {sr.composite_score:.1f}**\n",
                self._score_breakdown_table(sr),
                self._financials_table(sr),
            ]
        else:
            parts: list[str] = [
                f"### {idx}. {ca.company_name} ({ca.ticker})",
                f"**Composite Score: {sr.composite_score:.1f}**\n",
                self._score_breakdown_table_en(sr),
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
        if self.output_language == "zh-CN":
            return _zh_financials_table(sr)

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
        lines = [
            "#### Key Financials",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for label, val in rows:
            lines.append(f"| {label} | {val} |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _score_breakdown_table(sr) -> str:
        """Render the 6-factor score breakdown (Chinese)."""
        return (
            "#### 评分明细\n"
            "| 因子 | 权重 | 得分 |\n"
            "|------|------|------|\n"
            f"| 价值 (Value) | 40% | {sr.value_score:.1f} |\n"
            f"| 质量 (Quality) | 20% | {sr.quality_score:.1f} |\n"
            f"| 成长 (Growth) | 10% | {sr.growth_score:.1f} |\n"
            f"| 动量 (Momentum) | 10% | {sr.momentum_score:.1f} |\n"
            f"| 协同 (Synergy) | 10% | {sr.synergy_score:.1f} |\n"
            f"| 价值-成长 (Value-Growth) | 10% | {sr.value_growth_score:.1f} |\n"
            f"| **综合 (Composite)** | **100%** | **{sr.composite_score:.1f}** |\n"
        )

    @staticmethod
    def _score_breakdown_table_en(sr) -> str:
        """Render the 6-factor score breakdown (English)."""
        return (
            "#### Score Breakdown\n"
            "| Factor | Weight | Score |\n"
            "|--------|--------|-------|\n"
            f"| Value | 40% | {sr.value_score:.1f} |\n"
            f"| Quality | 20% | {sr.quality_score:.1f} |\n"
            f"| Growth | 10% | {sr.growth_score:.1f} |\n"
            f"| Momentum | 10% | {sr.momentum_score:.1f} |\n"
            f"| Synergy | 10% | {sr.synergy_score:.1f} |\n"
            f"| Value-Growth | 10% | {sr.value_growth_score:.1f} |\n"
            f"| **Composite** | **100%** | **{sr.composite_score:.1f}** |\n"
        )

    def _disclaimer(self) -> str:
        if self.output_language == "zh-CN":
            return _zh_disclaimer()

        return (
            "## Disclaimer\n\n"
            "**This report is NOT a stock forecast and does not constitute investment advice.**\n\n"
            "This is an algorithm-driven stock screening system based on 10 years "
            "(2016–2025) of historical financial data. The composite score is computed "
            "from value, quality, and growth factors via a quantitative model and "
            "reflects relative ranking across these dimensions only — **it does not "
            "predict future stock prices**.\n\n"
            "The Spearman ρ value measures the rank correlation between scores and "
            "historical actual returns, used for backtesting the scoring algorithm's "
            "validity. Past backtest results do not guarantee future performance.\n\n"
            "The information presented is derived from publicly available data and "
            "AI-driven analysis, which may contain inaccuracies or be out of date. "
            "Investors should conduct their own independent due diligence and consult "
            "a licensed professional financial advisor before making any investment "
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
        return (
            f"# {report.title}\n"
            f"*生成时间: {date_str}*\n\n"
            "> **重要提示：本报告不是股票预测。** "
            "这是一个基于过去10年（2016–2025）历史数据的算法驱动股票筛选系统。"
            "评分反映的是多因子量化排名，而非对未来股价的预测或投资建议。"
        )

    def _summary(self, report: MultiTimeframeReport, results: list) -> str:
        total_screened = report.total_screened
        n_candidates = len(results)
        lines = [
            "## 执行摘要\n",
            f"- 筛选公司总数: {total_screened}",
            f"- 入选候选数: {n_candidates} 只",
            f"- 个股分析覆盖: {len(report.company_analyses)} 只",
        ]
        # ρ summary — 6-month is the primary (Final) metric
        rhos = report.spearman_rhos
        if rhos:
            rho_6m = rhos.get("6m")
            if rho_6m is not None:
                quality = _rho_quality(rho_6m)
                lines.append(f"- **最终 ρ（6个月）: {rho_6m:.4f}**（{quality}）— 这是衡量评分算法有效性的核心指标")
            parts = []
            for h in ("1m", "3m", "6m"):
                lbl = _HORIZON_LABELS_ZH.get(h, h)
                rho = rhos.get(h)
                if rho is not None:
                    parts.append(f"{lbl} ρ={rho:.4f}")
            if parts:
                lines.append(f"- 各周期预测力: {' · '.join(parts)}")
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

        # Build the Final ρ (6-month) highlight
        rho_6m = rhos.get("6m") if rhos else None
        if rho_6m is not None:
            final_rho_section = (
                f"### 最终 ρ（6个月）: {rho_6m:.4f}\n\n"
                "**最终 ρ** 以 6 个月为预测周期，是衡量评分算法有效性的核心指标。"
                "它量化了综合评分排序与股票 6 个月后实际回报排序之间的一致性。\n\n"
                f"当前最终 ρ = **{rho_6m:.4f}**，评级为 **{_rho_quality(rho_6m)}**。"
                "这意味着：如果将所有股票按综合评分从高到低排列，"
                "评分越高的组别，其 6 个月后的实际回报排名也倾向于越高——"
                "评分系统具有超出随机选择的预测能力。\n"
            )
        else:
            final_rho_section = ""

        return f"""## 评分算法说明

### 本系统不是股票预测

**本报告中的评分和排名均不构成对未来股价的预测。**

这是一个基于 **2016–2025 年（过去10年）历史财务数据** 的算法驱动股票筛选系统。
综合评分通过多因子量化模型计算，仅反映企业在价值、质量、成长三个维度上
相对于所有同行的排序位置。评分的有效性通过 Spearman ρ（秩相关系数）进行
回测验证——即检查历史评分排序与事后实际回报排序之间的一致性。

### Spearman ρ（秩相关系数）

ρ 值衡量综合评分与未来实际回报的**排序一致性**。取值范围 -1 到 +1：

| ρ 范围 | 含义 |
|--------|------|
| **0.20 ~ 0.30** | 优秀 — 评分具有显著预测力，超出大多数单因子策略 |
| **0.10 ~ 0.20** | 良好 — 有一定预测力，但仍需改进 |
| **0.05 ~ 0.10** | 一般 — 仅略优于随机选择 |
| **0.00 ~ 0.05** | 极弱 — 几乎无预测力 |
| **< 0.00** | 反向 — 评分与回报负相关，需重新审视算法 |

> 在金融领域，单因子的 ρ 通常不超过 ±0.10。多因子模型能达到 0.15+
> 已属于优秀水平。ρ 为 0 表示评分与回报无任何单调关系。

{final_rho_section}
### 各周期 ρ 表现

| 预测周期 | Spearman ρ | 评级 |
|----------|-----------|------|
{rho_table}

### 综合评分（0–100）

综合评分由六个因子通过**加权调和平均**合成，权重严格对应 `scorer.py` 中的 `_DEFAULT_WEIGHTS`：

| 因子 | 权重 | 计算方式 |
|------|------|----------|
| **价值 (Value)** | 40% | PE、PB、PS、股息率、EV/EBITDA、盈利收益率、FCF收益率等 20+ 项子指标的加权平均，经幂次提升 |
| **质量 (Quality)** | 20% | `(ROE × 毛利率) / ((1 + sqrt(负债率)) × PE)` — 横截面百分位排名 |
| **成长 (Growth)** | 10% | PEG 倒数 — 横截面百分位排名 |
| **动量 (Momentum)** | 10% | 资产周转率（营收/总资产）— 效率指标 |
| **协同 (Synergy)** | 10% | `min(价值得分, 质量得分)` — 奖励两者均衡 |
| **价值-成长 (Value-Growth)** | 10% | `sqrt(价值得分 × 成长得分)` — 价值与成长的几何交互 |

调和平均的特性是：**任何一项因子得分极低都会严重拖累综合评分**，因此高分股票必须在所有维度上均表现良好。
综合评分还会乘以亏损惩罚系数（亏损企业按亏损/市值比例扣分）。

| 评分范围 | 含义 |
|----------|------|
| **60+** | 优秀 — 各项因子均衡且突出，是筛选系统能产生的最佳候选 |
| **50–60** | 良好 — 多数因子表现不错，综合吸引力较强 |
| **40–50** | 中等 — 部分因子存在短板 |
| **< 40** | 偏差 — 至少一项核心指标严重拖分 |

> 实际筛选结果通常在 50–65 分区间。100 分仅为理论最大值，现实中不存在
> 所有因子同时完美的股票。评分越高，历史上对应 6 个月后的实际回报排名
> 越倾向于靠前，但**这不保证任何特定股票的未来表现**。"""

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

        parts = [
            f"### {idx}. {ca.company_name} ({ca.ticker})",
            f"**综合评分: {sr.composite_score:.1f}**\n",
            self._score_breakdown_table(sr),
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

    @staticmethod
    def _score_breakdown_table(sr) -> str:
        """Render the 6-factor score breakdown matching scorer.py structure."""
        return (
            "#### 评分明细\n"
            "| 因子 | 权重 | 得分 |\n"
            "|------|------|------|\n"
            f"| 价值 (Value) | 40% | {sr.value_score:.1f} |\n"
            f"| 质量 (Quality) | 20% | {sr.quality_score:.1f} |\n"
            f"| 成长 (Growth) | 10% | {sr.growth_score:.1f} |\n"
            f"| 动量 (Momentum) | 10% | {sr.momentum_score:.1f} |\n"
            f"| 协同 (Synergy) | 10% | {sr.synergy_score:.1f} |\n"
            f"| 价值-成长 (Value-Growth) | 10% | {sr.value_growth_score:.1f} |\n"
            f"| **综合 (Composite)** | **100%** | **{sr.composite_score:.1f}** |\n"
        )

    def _financials_table(self, sr) -> str:
        return _zh_financials_table(sr)

    def _disclaimer(self) -> str:
        return _zh_disclaimer()


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
