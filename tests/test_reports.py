from __future__ import annotations

import json
import os
import time

from valueinvestor.data.models import (
    AnalysisDimension,
    Company,
    Financials,
    Market,
    MultiTimeframeReport,
    ScreeningResult,
    ValuationMetrics,
)
from valueinvestor.reports.md_generator import (
    MultiTimeframeMarkdownReportGenerator,
    _local_report_date,
    _render_analysis_dimension,
    _zh_upfront_model_notes,
)


def _screening_result() -> ScreeningResult:
    return ScreeningResult(
        company=Company(ticker="000001.SZ", name="Test Co", market=Market.A_SHARE),
        financials=Financials(ticker="000001.SZ", period="2024-12-31", roe=0.12),
        valuation=ValuationMetrics(
            ticker="000001.SZ",
            date="2026-05-06",
            pe_ratio=10.0,
            pb_ratio=1.2,
        ),
        composite_score=95.0,
        value_score=70.0,
        quality_score=65.0,
        growth_score=55.0,
        momentum_score=50.0,
        synergy_score=65.0,
        value_growth_score=62.0,
        rank=1,
    )


def test_report_date_converts_utc_timestamp_to_local_date(monkeypatch) -> None:
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    if hasattr(time, "tzset"):
        time.tzset()

    try:
        assert _local_report_date("2026-05-24T16:30:00+00:00") == "2026-05-25"
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        if hasattr(time, "tzset"):
            time.tzset()


def test_ml_report_notes_show_train_rho_without_holdout_rho() -> None:
    markdown = _zh_upfront_model_notes({
        "scorer_model": {
            "type": "ml_ranker",
            "backend": "mlx",
            "spearman_rhos": {"6m": 0.310611},
            "holdout_spearman_rhos": {"6m": 0.064608},
        }
    })

    assert "当前 6 个月训练/全量回测 Spearman ρ = **0.3106**" in markdown
    assert "未触碰持出集/推广门" not in markdown
    assert "0.0646" not in markdown
    assert "当前 6 个月 Spearman ρ = **0.0646**" not in markdown


def test_current_scorer_metadata_prefers_train_rho_over_holdout(tmp_path) -> None:
    from valueinvestor.cli.main import _load_current_scorer_metadata

    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({
            "schema_version": "test",
            "metadata": {
                "backend": "mlx",
                "target_horizon": "6m",
                "train_metrics": {"6m": {"spearman_rho": 0.310611}},
                "metrics": {"6m": {"spearman_rho": 0.064608}},
            },
        }),
        encoding="utf-8",
    )

    metadata = _load_current_scorer_metadata(model_path)

    assert metadata["spearman_rhos"]["6m"] == 0.310611
    assert metadata["train_spearman_rhos"]["6m"] == 0.310611
    assert metadata["holdout_spearman_rhos"]["6m"] == 0.064608
    assert metadata["spearman_rho_basis"] == "train_metrics"


def test_current_scorer_metadata_counts_payload_members(tmp_path) -> None:
    from valueinvestor.cli.main import _load_current_scorer_metadata

    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({
            "schema_version": "test",
            "payload_members": [
                {"weight": 0.9, "payload": {}},
                {"weight": 0.1, "payload": {}},
            ],
            "metadata": {
                "backend": "payload_member_blend",
                "target_horizon": "6m",
                "train_metrics": {"6m": {"spearman_rho": 0.309884}},
            },
        }),
        encoding="utf-8",
    )

    metadata = _load_current_scorer_metadata(model_path)

    assert metadata["ensemble_size"] == 2


def test_analysis_dimension_renderer_recovers_raw_json_fallback_text() -> None:
    raw = (
        '{"title": "管理层能力评估：特变电工的领导力", '
        '"content": "公司启动"零缺陷"质量体系，次年重获客户信任。", '
        '"confidence": 0.9}'
    )
    dim = AnalysisDimension(dimension="management", title="management", content=raw)

    markdown = _render_analysis_dimension(dim)

    assert "#### 管理层能力评估：特变电工的领导力" in markdown
    assert "公司启动\"零缺陷\"质量体系，次年重获客户信任。" in markdown
    assert '{"title":' not in markdown


def test_multi_timeframe_report_describes_active_ml_ranker() -> None:
    result = _screening_result()
    report = MultiTimeframeReport(
        title="价值投资多时间框架筛选报告",
        generated_at="2026-05-06T00:00:00+00:00",
        config_summary={
            "scorer_model": {
                "type": "ml_ranker",
                "backend": "ensemble",
                "ensemble_size": 3,
            }
        },
        total_screened=100,
        results_by_horizon={"6m": [result]},
        company_analyses=[],
        spearman_rhos={"1m": 0.11, "3m": 0.19, "6m": 0.2253},
    )

    markdown = MultiTimeframeMarkdownReportGenerator().generate(report)

    assert "ML 排名器" in markdown
    assert "综合评分（ML百分位）" in markdown
    assert "加权调和平均" not in markdown


def test_multi_timeframe_report_uses_scorer_model_rho_fallback() -> None:
    result = _screening_result()
    report = MultiTimeframeReport(
        title="价值投资多时间框架筛选报告",
        generated_at="2026-05-06T00:00:00+00:00",
        config_summary={
            "scorer_model": {
                "type": "ml_ranker",
                "backend": "ensemble",
                "ensemble_size": 3,
                "metrics": {
                    "6m": {"spearman_rho": 0.2334},
                },
            }
        },
        total_screened=100,
        results_by_horizon={"6m": [result]},
        company_analyses=[],
        spearman_rhos={},
    )

    markdown = MultiTimeframeMarkdownReportGenerator().generate(report)

    assert "0.2334" in markdown
    assert "最终 ρ（6个月）: 0.2334" in markdown


def test_dual_target_report_renders_one_week_and_six_month_tables() -> None:
    one_week = _screening_result()
    one_week.company.ticker = "000001.SZ"
    six_month = _screening_result()
    six_month.company.ticker = "000002.SZ"
    report = MultiTimeframeReport(
        title="价值投资双目标筛选报告",
        generated_at="2026-05-17T00:00:00+00:00",
        config_summary={
            "scorer_models": {
                "1w": {"type": "ml_ranker", "backend": "numpy", "target_horizon": "1w"},
                "6m": {"type": "ml_ranker", "backend": "ensemble", "target_horizon": "6m"},
            },
            "score_targets": ["1w", "6m"],
        },
        total_screened=100,
        results_by_horizon={"1w": [one_week], "6m": [six_month]},
        company_analyses=[],
        spearman_rhos={"1w": 0.0412, "6m": 0.2253},
    )

    markdown = MultiTimeframeMarkdownReportGenerator().generate(report)

    assert "1周目标 Top 30" in markdown
    assert "6个月目标 Top 30" in markdown
    assert "1周ML百分位" in markdown
    assert "6个月ML百分位" in markdown
    assert "1周 | 0.0412" in markdown
    assert "6个月 | 0.2253" in markdown
