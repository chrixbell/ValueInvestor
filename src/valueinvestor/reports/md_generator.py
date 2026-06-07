"""Markdown report generator for China value-investment reports."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from valueinvestor.data.models import CompanyAnalysis, InvestmentReport, MultiTimeframeReport


_RAW_JSON_DIMENSION_RE = re.compile(
    r'^\s*\{\s*"title"\s*:\s*"(?P<title>.*?)"\s*,\s*"content"\s*:\s*"(?P<content>.*)"\s*,\s*"confidence"\s*:',
    re.DOTALL,
)


def _decode_raw_json_fragment(value: str) -> str:
    return value.replace("\\n", "\n").replace('\\"', '"').strip()


def _normalise_analysis_dimension_text(dim) -> tuple[str, str]:
    """Recover title/content when a cached LLM fallback is JSON-shaped text."""
    title = str(dim.title)
    content = str(dim.content)
    stripped = content.strip()
    if not (stripped.startswith("{") and '"title"' in stripped and '"content"' in stripped):
        return title, content

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _RAW_JSON_DIMENSION_RE.match(stripped)
        if not match:
            return title, content
        return (
            _decode_raw_json_fragment(match.group("title")),
            _decode_raw_json_fragment(match.group("content")),
        )

    if not isinstance(parsed, dict):
        return title, content
    parsed_title = parsed.get("title")
    parsed_content = parsed.get("content")
    if parsed_title is not None:
        title = str(parsed_title).strip()
    if parsed_content is not None:
        content = str(parsed_content).strip()
    return title, content


def _render_analysis_dimension(dim) -> str:
    title, content = _normalise_analysis_dimension_text(dim)
    return f"#### {title}\n{content}\n"


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


def _local_report_date(generated_at: str | None) -> str:
    """Return a local calendar date for an ISO report timestamp."""
    if not generated_at:
        return "N/A"
    try:
        parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return generated_at[:10]
    return parsed.astimezone().strftime("%Y-%m-%d")


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
    candidates = []
    if isinstance(rhos, dict):
        candidates.append(rhos)

    scorer_model = cfg.get("scorer_model")
    if isinstance(scorer_model, dict):
        nested_rhos = scorer_model.get("spearman_rhos")
        if isinstance(nested_rhos, dict):
            candidates.append(nested_rhos)
        nested_metrics = scorer_model.get("metrics")
        if isinstance(nested_metrics, dict):
            candidates.append(nested_metrics)

    for candidate in candidates:
        rho = candidate.get("6m")
        if isinstance(rho, dict):
            rho = rho.get("spearman_rho")
        if rho is None:
            continue
        try:
            return float(rho)
        except (TypeError, ValueError):
            continue
    return None


def _holdout_rho_value_from_config(cfg: dict) -> float | None:
    scorer_model = _scorer_model_from_config(cfg)
    rhos = scorer_model.get("holdout_spearman_rhos")
    if not isinstance(rhos, dict):
        return None
    rho = rhos.get("6m")
    try:
        return None if rho is None else float(rho)
    except (TypeError, ValueError):
        return None


def _report_spearman_rhos(report: MultiTimeframeReport) -> dict[str, float]:
    """Return the best available Spearman ρ snapshot for a report."""
    rhos = report.spearman_rhos or {}
    if isinstance(rhos, dict) and rhos:
        resolved: dict[str, float] = {}
        for horizon in ("1w", "1m", "3m", "6m"):
            rho = rhos.get(horizon)
            if rho is None:
                continue
            try:
                resolved[horizon] = float(rho)
            except (TypeError, ValueError):
                continue
        if resolved:
            return resolved

    cfg = report.config_summary or {}
    scorer_model = cfg.get("scorer_model")
    if isinstance(scorer_model, dict):
        nested_rhos = scorer_model.get("spearman_rhos")
        if isinstance(nested_rhos, dict) and nested_rhos:
            return {
                horizon: float(rho)
                for horizon, rho in nested_rhos.items()
                if horizon in ("1w", "1m", "3m", "6m") and rho is not None
                and isinstance(rho, (int, float))
            }
        nested_metrics = scorer_model.get("metrics")
        if isinstance(nested_metrics, dict):
            resolved = {}
            for horizon in ("1w", "1m", "3m", "6m"):
                horizon_metrics = nested_metrics.get(horizon)
                if not isinstance(horizon_metrics, dict):
                    continue
                rho = horizon_metrics.get("spearman_rho")
                if rho is None:
                    continue
                try:
                    resolved[horizon] = float(rho)
                except (TypeError, ValueError):
                    continue
            if resolved:
                return resolved
    return {}


def _scorer_model_from_config(cfg: dict) -> dict:
    scorer_model = cfg.get("scorer_model")
    return scorer_model if isinstance(scorer_model, dict) else {}


def _uses_ml_ranker(cfg: dict) -> bool:
    if _scorer_model_from_config(cfg).get("type") == "ml_ranker":
        return True
    scorer_models = cfg.get("scorer_models")
    if not isinstance(scorer_models, dict):
        return False
    return any(
        isinstance(model, dict) and model.get("type") == "ml_ranker"
        for model in scorer_models.values()
    )


def _ml_ranker_label(cfg: dict) -> str:
    scorer_model = _scorer_model_from_config(cfg)
    backend = scorer_model.get("backend") or "local"
    ensemble_size = scorer_model.get("ensemble_size")
    if ensemble_size:
        return f"{backend}, ensemble_size={ensemble_size}"
    return str(backend)


def _zh_upfront_model_notes(cfg: dict) -> str:
    """Render the three upfront model, rho, and score explanation sections."""
    rho_6m = _rho_value_from_config(cfg)
    if rho_6m is None:
        rho_current = "当前报告未附带可用的 6 个月 Spearman ρ 数值。"
    else:
        rho_current = (
            f"当前 6 个月训练/全量回测 Spearman ρ = **{rho_6m:.4f}**，"
            f"评级为 **{_rho_quality(rho_6m)}**。"
        )
    if _uses_ml_ranker(cfg):
        score_note = (
            f"当前启用本地 ML 排名器（{_ml_ranker_label(cfg)}）。综合评分是模型原始排序分数"
            "在本轮候选中的 0–100 百分位；价值、质量、成长、动量等手工因子仍作为模型输入"
            "和诊断项展示，但不再是最终分数的简单加权合成。"
        )
    else:
        score_note = (
            "综合评分是 0–100 的相对评分，用来比较同一轮筛选中公司基本面、估值、质量、成长、动量和因子均衡性的综合表现。"
            "分数越高，表示该公司在模型历史回测框架下越接近未来 6 个月收益排序靠前的特征，"
            "但这**不保证**任何单一股票未来上涨。"
        )

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
        f"{score_note}\n\n"
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
        rho_current = (
            f"Current 6-month training/full-backtest Spearman rho = **{rho_6m:.4f}**."
        )
    if _uses_ml_ranker(cfg):
        score_note = (
            f"The active scorer is a local ML ranker ({_ml_ranker_label(cfg)}). "
            "The final composite score is the 0-100 percentile of the model's raw rank score "
            "within the current candidate set. The hand factor scores remain inputs and diagnostics, "
            "but they are not the final score's direct weighted formula."
        )
    else:
        score_note = (
            "The composite score is a 0-100 relative score. Higher scores mean the company better "
            "matches characteristics that historically ranked better over the following 6 months, "
            "but this does not guarantee future performance."
        )

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
        f"{score_note}\n"
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
        self._ml_ranker_active = _uses_ml_ranker(cfg)

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
        date_str = _local_report_date(report.generated_at)
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
            parts.append(_render_analysis_dimension(dim))

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

    def _score_breakdown_table(self, sr) -> str:
        """Render the 6-factor score breakdown (Chinese)."""
        if getattr(self, "_ml_ranker_active", False):
            return (
                "#### 评分明细\n"
                "| 项目 | 含义 | 得分 |\n"
                "|------|------|------|\n"
                f"| **综合评分** | **ML 排名器百分位** | **{sr.composite_score:.1f}** |\n"
                f"| 价值 (Value) | 模型输入/诊断项 | {sr.value_score:.1f} |\n"
                f"| 质量 (Quality) | 模型输入/诊断项 | {sr.quality_score:.1f} |\n"
                f"| 成长 (Growth) | 模型输入/诊断项 | {sr.growth_score:.1f} |\n"
                f"| 动量 (Momentum) | 模型输入/诊断项 | {sr.momentum_score:.1f} |\n"
                f"| 协同 (Synergy) | 模型输入/诊断项 | {sr.synergy_score:.1f} |\n"
                f"| 价值-成长 (Value-Growth) | 模型输入/诊断项 | {sr.value_growth_score:.1f} |\n"
            )
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

    def _score_breakdown_table_en(self, sr) -> str:
        """Render the 6-factor score breakdown (English)."""
        if getattr(self, "_ml_ranker_active", False):
            return (
                "#### Score Breakdown\n"
                "| Item | Meaning | Score |\n"
                "|------|---------|-------|\n"
                f"| **Composite Score** | **ML ranker percentile** | **{sr.composite_score:.1f}** |\n"
                f"| Value | Model input / diagnostic | {sr.value_score:.1f} |\n"
                f"| Quality | Model input / diagnostic | {sr.quality_score:.1f} |\n"
                f"| Growth | Model input / diagnostic | {sr.growth_score:.1f} |\n"
                f"| Momentum | Model input / diagnostic | {sr.momentum_score:.1f} |\n"
                f"| Synergy | Model input / diagnostic | {sr.synergy_score:.1f} |\n"
                f"| Value-Growth | Model input / diagnostic | {sr.value_growth_score:.1f} |\n"
            )
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
    "1w": "1周",
    "1m": "1个月",
    "3m": "3个月",
    "6m": "6个月",
}

_TARGET_ORDER = ("1w", "6m", "1m", "3m")


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
        self._ml_ranker_active = _uses_ml_ranker(report.config_summary or {})
        self._spearman_rhos = _report_spearman_rhos(report)
        target_order = self._target_order(report)
        if self._is_combined_target_report(report, target_order):
            sections = [
                self._heading(report),
                self._combined_summary(report, target_order),
                self._combined_scoring_explanation(report, target_order),
                self._combined_candidates_tables(report, target_order),
                self._detailed_analyses(report),
                self._disclaimer(),
            ]
            return "\n\n".join(sections)

        results = report.results_by_horizon.get("6m", []) or next(iter(report.results_by_horizon.values()), [])
        sections = [
            self._heading(report),
            self._summary(report, results),
            self._scoring_explanation(report),
        ]
        sections.append(self._candidates_table(results, report.config_summary))
        sections.append(self._detailed_analyses(report))
        sections.append(self._disclaimer())
        return "\n\n".join(sections)

    def save(self, report: MultiTimeframeReport, output_dir: str = "reports") -> str:
        """Generate and save the report; return the saved file path."""
        md_content = self.generate(report)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        target_order = self._target_order(report)
        if len(target_order) > 1 and self._is_combined_target_report(report, target_order):
            suffix = "dual_target"
        elif target_order:
            suffix = f"{target_order[0]}_target"
        else:
            suffix = "multi_timeframe"
        filepath = out / f"{date_str}_china_value_{suffix}.md"
        filepath.write_text(md_content, encoding="utf-8")
        return str(filepath)

    def _target_order(self, report: MultiTimeframeReport) -> list[str]:
        keys = set(report.results_by_horizon)
        ordered = [target for target in _TARGET_ORDER if target in keys]
        ordered.extend(sorted(keys.difference(ordered)))
        return ordered

    def _is_combined_target_report(
        self,
        report: MultiTimeframeReport,
        target_order: list[str],
    ) -> bool:
        cfg = report.config_summary or {}
        return bool(target_order) and ("1w" in target_order or bool(cfg.get("scorer_models")))

    def _heading(self, report: MultiTimeframeReport) -> str:
        date_str = _local_report_date(report.generated_at)
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
        rhos = getattr(self, "_spearman_rhos", {})
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

    def _combined_summary(self, report: MultiTimeframeReport, target_order: list[str]) -> str:
        lines = [
            "## 执行摘要\n",
            f"- 筛选公司总数: {report.total_screened}",
            f"- 个股分析覆盖: {len(report.company_analyses)} 只",
        ]
        for target in target_order:
            label = _HORIZON_LABELS_ZH.get(target, target)
            results = report.results_by_horizon.get(target, [])
            rho = getattr(self, "_spearman_rhos", {}).get(target)
            if rho is None:
                lines.append(f"- {label}入选候选数: {len(results)} 只")
            else:
                lines.append(
                    f"- {label}入选候选数: {len(results)} 只；"
                    f"Spearman ρ={rho:.4f}（{_rho_quality(rho)}）"
                )
        return "\n".join(lines)

    def _combined_scoring_explanation(
        self,
        report: MultiTimeframeReport,
        target_order: list[str],
    ) -> str:
        rhos = getattr(self, "_spearman_rhos", {})
        rho_lines = []
        for target in target_order:
            label = _HORIZON_LABELS_ZH.get(target, target)
            rho = rhos.get(target)
            if rho is None:
                rho_lines.append(f"| {label} | 暂无数据 | — |")
            else:
                rho_lines.append(f"| {label} | {rho:.4f} | {_rho_quality(rho)} |")

        return f"""## 评分算法说明

### 本系统不是股票预测

本报告中的评分和排名均不构成对未来股价的预测。两个榜单使用同一批筛选后的候选股票、同一套点时间财务/估值特征，但分别输入不同目标周期训练出的 ML 排名器。

### Spearman ρ（秩相关系数）

ρ 衡量模型评分排序与历史实际回报排序的一致性。1周模型使用 5 个交易日后的回报排序训练；6个月模型使用约 126 个自然日后的回报排序训练。

| 评分目标 | Spearman ρ | 评级 |
|----------|-----------|------|
{chr(10).join(rho_lines)}

### 综合评分（0–100）

每个榜单的综合评分都是对应 ML 排名器原始排序分数在本轮候选集中的 0–100 百分位。价值、质量、成长、动量等手工因子仍作为模型输入和诊断项展示，但最终排名以对应目标模型的 ML 百分位为准。

| 评分范围 | 含义 |
|----------|------|
| **80+** | 本轮候选中排名非常靠前 |
| **60–80** | 本轮候选中排名靠前 |
| **40–60** | 本轮候选中居中 |
| **< 40** | 本轮候选中排名靠后 |

> 分数是横截面相对分，不是上涨概率，也不是目标收益率。"""

    def _combined_candidates_tables(
        self,
        report: MultiTimeframeReport,
        target_order: list[str],
    ) -> str:
        sections = ["## 最佳投资标的\n"]
        for target in target_order:
            label = _HORIZON_LABELS_ZH.get(target, target)
            sections.append(
                self._candidates_table(
                    report.results_by_horizon.get(target, []),
                    report.config_summary,
                    title=f"### {label}目标 Top 30",
                    score_header=f"{label}ML百分位",
                )
            )
        return "\n\n".join(sections)

    def _scoring_explanation(self, report: MultiTimeframeReport) -> str:
        """Explain ρ values, ρ interpretation, and score interpretation."""
        rhos = getattr(self, "_spearman_rhos", {})
        cfg = report.config_summary or {}

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

        if _uses_ml_ranker(cfg):
            score_section = f"""### 综合评分（0–100）

当前报告启用本地 ML 排名器（{_ml_ranker_label(cfg)}）。综合评分不是手工因子的直接加权结果，而是：

1. 先计算基础手工因子得分与财务/估值派生特征；
2. 将这些特征输入当前最佳 ML 排名器，得到每只股票的原始排序分数；
3. 按原始排序分数从低到高转换为本轮候选集内的 0–100 百分位分数。

因此，报告中的价值、质量、成长、动量等因子仍然有用：它们解释模型看到的企业特征，也帮助定位候选股票的优势和短板。但最终排名以 ML 排名器输出的综合评分为准。

| 评分范围 | 含义 |
|----------|------|
| **80+** | 本轮候选中排名非常靠前 |
| **60–80** | 本轮候选中排名靠前 |
| **40–60** | 本轮候选中居中 |
| **< 40** | 本轮候选中排名靠后 |

> 分数是横截面相对分，不是上涨概率，也不是目标收益率。"""
        else:
            score_section = """### 综合评分（0–100）

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
| **< 40** | 偏差 — 至少一项核心指标严重拖分 |"""

        if _uses_ml_ranker(cfg):
            score_tail = (
                "> 评分越高，历史上对应 6 个月后的实际回报排名越倾向于靠前，"
                "但**这不保证任何特定股票的未来表现**。"
            )
        else:
            score_tail = (
                "> 实际筛选结果通常在 50–65 分区间。100 分仅为理论最大值，现实中不存在\n"
                "> 所有因子同时完美的股票。评分越高，历史上对应 6 个月后的实际回报排名\n"
                "> 越倾向于靠前，但**这不保证任何特定股票的未来表现**。"
            )

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

{score_section}

{score_tail}"""

    def _candidates_table(
        self,
        results: list,
        cfg: dict,
        *,
        title: str = "## 最佳投资标的",
        score_header: str | None = None,
    ) -> str:
        if score_header is None:
            score_header = "综合评分（ML百分位）" if _uses_ml_ranker(cfg) else "综合评分"
        lines = [
            f"{title}\n",
            f"| 排名 | 代码 | 公司 | 行业 | {score_header} |",
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
            parts.append(_render_analysis_dimension(dim))

        if ca.key_milestones:
            parts.append("**需要关注的关键里程碑：**")
            for ms in ca.key_milestones:
                parts.append(f"- {ms}")
            parts.append("")

        parts.append("---\n")
        return "\n".join(parts)

    def _score_breakdown_table(self, sr) -> str:
        """Render the 6-factor score breakdown matching scorer.py structure."""
        if getattr(self, "_ml_ranker_active", False):
            return (
                "#### 评分明细\n"
                "| 项目 | 含义 | 得分 |\n"
                "|------|------|------|\n"
                f"| **综合评分** | **ML 排名器百分位** | **{sr.composite_score:.1f}** |\n"
                f"| 价值 (Value) | 模型输入/诊断项 | {sr.value_score:.1f} |\n"
                f"| 质量 (Quality) | 模型输入/诊断项 | {sr.quality_score:.1f} |\n"
                f"| 成长 (Growth) | 模型输入/诊断项 | {sr.growth_score:.1f} |\n"
                f"| 动量 (Momentum) | 模型输入/诊断项 | {sr.momentum_score:.1f} |\n"
                f"| 协同 (Synergy) | 模型输入/诊断项 | {sr.synergy_score:.1f} |\n"
                f"| 价值-成长 (Value-Growth) | 模型输入/诊断项 | {sr.value_growth_score:.1f} |\n"
            )
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
