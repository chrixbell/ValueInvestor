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


def test_ml_report_notes_show_holdout_rho_when_available() -> None:
    markdown = _zh_upfront_model_notes({
        "scorer_model": {
            "type": "ml_ranker",
            "backend": "mlx",
            "spearman_rhos": {"6m": 0.310611},
            "holdout_spearman_rhos": {"6m": 0.064608},
            "promotion_status": "provisional",
            "promotion_gate_min_primary_rho": 0.08,
        }
    })

    assert "当前 6 个月未触碰持出集 Spearman ρ = **0.0646**" in markdown
    assert "0.3106" not in markdown
    assert "训练/全量回测 Spearman ρ" not in markdown
    assert "验证状态：临时研究模型" in markdown
    assert "预设绝对推广门槛 0.0800" in markdown


def test_current_scorer_metadata_prefers_holdout_rho_over_train(tmp_path) -> None:
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
                "promotion_status": "provisional",
                "promotion_gate": {
                    "accepted": False,
                    "reason": "6m rho below absolute gate",
                    "config": {"min_primary_rho": 0.08},
                },
            },
        }),
        encoding="utf-8",
    )

    metadata = _load_current_scorer_metadata(model_path)

    assert metadata["spearman_rhos"]["6m"] == 0.064608
    assert metadata["train_spearman_rhos"]["6m"] == 0.310611
    assert metadata["holdout_spearman_rhos"]["6m"] == 0.064608
    assert metadata["spearman_rho_basis"] == "holdout_metrics"
    assert metadata["promotion_status"] == "provisional"
    assert metadata["promotion_gate_accepted"] is False
    assert metadata["promotion_gate_reason"] == "6m rho below absolute gate"
    assert metadata["promotion_gate_min_primary_rho"] == 0.08


def test_current_scorer_metadata_prefers_refit_validation_record(tmp_path) -> None:
    from valueinvestor.cli.main import _load_current_scorer_metadata

    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({
            "schema_version": "test",
            "metadata": {
                "backend": "mlx",
                "target_horizon": "6m",
                "metrics": {"6m": {"spearman_rho": 0.192264}},
                "validation_metrics": {"6m": {"spearman_rho": 0.093042}},
                "validation_promotion_gate": {
                    "accepted": True,
                    "reason": "accepted",
                    "config": {"min_primary_rho": 0.08},
                },
            },
        }),
        encoding="utf-8",
    )

    metadata = _load_current_scorer_metadata(model_path)

    assert metadata["spearman_rhos"]["6m"] == 0.093042
    assert metadata["holdout_spearman_rhos"]["6m"] == 0.093042
    assert metadata["promotion_gate_accepted"] is True
    assert metadata["promotion_gate_min_primary_rho"] == 0.08


def test_full_refit_scorer_metadata_is_not_labeled_as_holdout(tmp_path) -> None:
    from valueinvestor.cli.main import _load_current_scorer_metadata

    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({
            "schema_version": "test",
            "rank_payload_members": [
                {"weight": 0.625, "payload": {}},
                {"weight": 0.375, "payload": {}},
            ],
            "metadata": {
                "backend": "rank_payload_blend",
                "model_kind": "lightgbm_incumbent_rank_blend",
                "target_horizon": "6m",
                "evaluation_protocol": "full_refit_current_24m_monthly_gate",
                "strict_outer_gate": False,
                "promotion_status": "target_met_full_refit_purged_audit_failed",
                "strict_purged_audit": {
                    "accepted": False,
                    "spearman_rho": 0.033618,
                    "best_spearman_rho": 0.084930,
                },
                "metrics": {"6m": {"spearman_rho": 0.268024}},
            },
        }),
        encoding="utf-8",
    )

    metadata = _load_current_scorer_metadata(model_path)

    assert metadata["spearman_rhos"]["6m"] == 0.268024
    assert "holdout_spearman_rhos" not in metadata
    assert metadata["spearman_rho_basis"] == "full_refit_current_24m_monthly_gate"
    assert metadata["ensemble_size"] == 2
    assert metadata["model_kind"] == "lightgbm_incumbent_rank_blend"
    assert metadata["strict_outer_gate"] is False
    assert metadata["strict_purged_audit"]["accepted"] is False


def test_full_refit_report_discloses_failed_strict_purged_audit() -> None:
    markdown = _zh_upfront_model_notes({
        "scorer_model": {
            "type": "ml_ranker",
            "backend": "rank_payload_blend",
            "model_kind": "lightgbm_incumbent_rank_blend",
            "ensemble_size": 2,
            "spearman_rhos": {"6m": 0.268024},
            "spearman_rho_basis": "full_refit_current_24m_monthly_gate",
            "promotion_status": "target_met_full_refit_purged_audit_failed",
            "strict_purged_audit": {
                "accepted": False,
                "spearman_rho": 0.033618,
                "best_spearman_rho": 0.084930,
            },
        }
    })

    assert "全量重拟合/当前 24 个月月度评估 Spearman ρ = **0.2680**" in markdown
    assert "未触碰持出集 Spearman ρ = **0.2680**" not in markdown
    assert "严格时间净化审计未通过" in markdown
    assert "不是未触碰持出集结果" in markdown
    assert "**0.0336**" in markdown
    assert "**0.0849**" in markdown
    assert "lightgbm_incumbent_rank_blend" in markdown
    assert "ensemble_size=2" in markdown


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
