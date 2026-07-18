"""Tests for the scorer_improver sub-package.

Tests cover experiment_log, evaluator, agent, and ground_truth helpers.
"""

from __future__ import annotations

import json
import time
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def test_lightgbm_payload_prediction_matches_runtime_model() -> None:
    import lightgbm as lgb

    from valueinvestor.scorer_improver.ml_trainer import (
        build_feature_matrix,
        predict_ml_ranker_payload,
    )
    from valueinvestor.screener.ml_ranker import (
        MODEL_SCHEMA_VERSION,
        _model_from_payload,
        expected_feature_names,
    )

    feature_names = expected_feature_names()
    frame = pd.DataFrame({
        "ticker": [f"T{index}" for index in range(8)],
        "snapshot_date": ["2025-01-02"] * 8,
        "value_score": np.linspace(10.0, 80.0, 8),
    })
    features = build_feature_matrix(frame, {}, feature_names=feature_names)
    target = np.linspace(-1.0, 1.0, len(frame))
    booster = lgb.train(
        {
            "objective": "regression_l2",
            "verbosity": -1,
            "num_threads": 1,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "feature_pre_filter": False,
        },
        lgb.Dataset(features, label=target, feature_name=feature_names),
        num_boost_round=3,
    )
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": feature_names,
        "lightgbm_model": booster.model_to_string(),
        "ticker_priors": {},
        "metadata": {},
    }

    offline = predict_ml_ranker_payload(frame, payload)
    runtime = _model_from_payload(payload).predict_matrix(features)

    assert offline.tolist() == pytest.approx(runtime.tolist())


def test_rank_payload_member_blend_matches_offline_and_live_runtime(tmp_path: Path) -> None:
    from valueinvestor.data.models import (
        Company,
        Financials,
        Market,
        ScreeningResult,
        ValuationMetrics,
    )
    from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload
    from valueinvestor.screener.ml_ranker import (
        MODEL_SCHEMA_VERSION,
        clear_model_cache,
        expected_feature_names,
        score_results_with_ml_ranker,
    )

    feature_names = expected_feature_names()

    def member_payload(direction: float) -> dict[str, object]:
        coefficients = [0.0] * len(feature_names)
        coefficients[feature_names.index("value_score")] = direction
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coefficients,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {},
        }

    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": [],
        "rank_payload_members": [
            {"weight": 0.75, "payload": member_payload(1.0)},
            {"weight": 0.25, "payload": member_payload(-1.0)},
        ],
        "ticker_priors": {},
        "metadata": {},
    }
    frame = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"],
        "snapshot_date": ["2025-01-02"] * 3,
        "value_score": [10.0, 20.0, 30.0],
    })
    offline = predict_ml_ranker_payload(frame, payload)
    assert offline.tolist() == pytest.approx([-0.25, 0.0, 0.25])

    model_path = tmp_path / "rank_blend.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    results = [
        ScreeningResult(
            company=Company(ticker=ticker, name=ticker, market=Market.A_SHARE),
            financials=Financials(ticker=ticker, period="snapshot"),
            valuation=ValuationMetrics(ticker=ticker, date="2025-01-02", price=1.0),
            value_score=value_score,
            composite_score=50.0,
        )
        for ticker, value_score in zip(frame["ticker"], frame["value_score"])
    ]
    clear_model_cache()
    assert score_results_with_ml_ranker(results, model_path)
    live = [float(result._ml_ranker_raw_score) for result in results]

    assert live == pytest.approx(offline.tolist())


def test_training_data_complete_requires_current_fetch_date(tmp_path: Path) -> None:
    from valueinvestor.scorer_improver.data_prep import _training_data_complete

    prices = tmp_path / "prices.parquet"
    prices.write_text("stub", encoding="utf-8")

    kwargs = {
        "force": False,
        "stored_start": "20160714",
        "required_start": "20160714",
        "required_fetch_date": "2026-07-12",
        "required_files": (prices,),
    }

    assert not _training_data_complete(last_fetch="2026-07-11", **kwargs)
    assert _training_data_complete(last_fetch="2026-07-12", **kwargs)


def test_hk_price_refresh_keeps_tickers_from_existing_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import yfinance as yf

    from valueinvestor.data.models import Company, Market
    from valueinvestor.scorer_improver import data_prep

    prices_path = tmp_path / "hkshare_prices.parquet"
    pd.DataFrame(
        {
            "ticker": ["0002.HK"],
            "date": [pd.Timestamp("2026-01-01")],
            "open": [10.0],
            "close": [10.0],
            "high": [10.0],
            "low": [10.0],
            "volume": [100.0],
        }
    ).to_parquet(prices_path, index=False)
    calls: list[str] = []

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, *, start: str, end: str) -> pd.DataFrame:
            del start, end
            calls.append(self.ticker)
            return pd.DataFrame(
                {
                    "Open": [11.0],
                    "Close": [11.0],
                    "High": [11.0],
                    "Low": [11.0],
                    "Volume": [110.0],
                },
                index=pd.DatetimeIndex(["2026-01-20"], name="Date"),
            )

    monkeypatch.setattr(data_prep, "_HKSHARE_PRICES_FILE", prices_path)
    monkeypatch.setattr(data_prep, "_YF_WORKERS", 1)
    monkeypatch.setattr(data_prep, "_SAVE_INTERVAL", 1_000)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)

    result = data_prep._fetch_hkshare_prices(
        [Company(ticker="0001.HK", name="New", market=Market.HK_SHARE)],
        "20260101",
        "20260120",
    )

    assert set(calls) == {"0001.HK", "0002.HK"}
    assert set(result["ticker"]) == {"0001.HK", "0002.HK"}


def test_latest_supported_daily_end_date_uses_calendar_horizon() -> None:
    from valueinvestor.scorer_improver.ground_truth import FORWARD_HORIZON_6M_DAYS
    from valueinvestor.scorer_improver.ml_trainer import (
        _latest_supported_daily_end_date_from_prices,
    )

    ashare_dates = pd.bdate_range("2025-12-01", periods=140)
    hkshare_dates = pd.bdate_range("2025-12-03", periods=135)
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "000001.SZ", "date": ashare_dates}),
            pd.DataFrame({"ticker": "0001.HK", "date": hkshare_dates}),
        ],
        ignore_index=True,
    )

    resolved = _latest_supported_daily_end_date_from_prices(prices)
    expected_by_market = []
    for dates in (ashare_dates, hkshare_dates):
        cutoff = dates[-1] - pd.Timedelta(days=FORWARD_HORIZON_6M_DAYS)
        expected_by_market.append(dates[dates <= cutoff][-1].date())
    expected = min(expected_by_market)

    assert resolved == expected


def test_six_month_target_is_182_calendar_days() -> None:
    from valueinvestor.scorer_improver.ground_truth import (
        FORWARD_HORIZON_6M_DAYS,
        _compute_forward_returns,
    )

    assert FORWARD_HORIZON_6M_DAYS == 182
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": ["2024-01-01", "2024-05-06", "2024-07-01"],
            "close": [10.0, 20.0, 30.0],
        }
    )

    result = _compute_forward_returns(
        prices,
        horizon_days=FORWARD_HORIZON_6M_DAYS,
    ).set_index("date")

    assert result.loc[pd.Timestamp("2024-01-01").date(), "forward_return_6m"] == pytest.approx(2.0)


def test_application_universe_mask_matches_production_filters() -> None:
    from valueinvestor.scorer_improver.ml_trainer import _application_universe_mask

    frame = pd.DataFrame(
        {
            "market_cap_rmb": [5e9, 4.9e9, 5e9, 5e9, 5e9, 5e9, 5e9],
            "pe_ratio": [30.0, 10.0, 0.0, 31.0, 10.0, 10.0, 10.0],
            "pb_ratio": [5.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            "roe": [0.08, 0.20, 0.20, 0.20, 0.20, 0.079, 0.20],
            "debt_to_equity": [None, None, None, None, None, None, 0.71],
        }
    )

    assert _application_universe_mask(frame).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_application_universe_rejects_missing_valuation_history() -> None:
    from valueinvestor.scorer_improver.ml_trainer import (
        _filter_application_training_universe,
    )

    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "snapshot_date": ["2024-01-31", "2024-01-31"],
            "market_cap_rmb": [10e9, 10e9],
            "pe_ratio": [None, None],
            "pb_ratio": [None, None],
            "roe": [0.2, 0.2],
            "debt_to_equity": [0.2, 0.2],
        }
    )

    with pytest.raises(RuntimeError, match="cannot represent the production stock screen"):
        _filter_application_training_universe(
            frame,
            min_rows_per_snapshot=1,
            min_snapshots=1,
        )


def test_application_universe_reranks_after_filtering() -> None:
    from valueinvestor.scorer_improver.ml_trainer import (
        _filter_application_training_universe,
    )

    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "snapshot_date": ["2024-01-31"] * 3,
            "market_cap_rmb": [10e9] * 3,
            "pe_ratio": [10.0, 20.0, 40.0],
            "pb_ratio": [1.0, 2.0, 1.0],
            "roe": [0.2, 0.2, 0.2],
            "debt_to_equity": [0.2, 0.2, 0.2],
            "forward_return_1w": [0.1, 0.3, 0.9],
            "forward_return_1m": [0.1, 0.3, 0.9],
            "forward_return_3m": [0.1, 0.3, 0.9],
            "forward_return_6m": [0.1, 0.3, 0.9],
        }
    )

    filtered = _filter_application_training_universe(
        frame,
        min_rows_per_snapshot=2,
        min_snapshots=1,
    )

    assert filtered["ticker"].tolist() == ["AAA", "BBB"]
    assert filtered["target_rank_6m"].tolist() == pytest.approx([-1 / 3, 1 / 3])


def test_monthly_rebalance_uses_well_populated_snapshot() -> None:
    from valueinvestor.scorer_improver.ml_trainer import _monthly_rebalance_snapshots

    frame = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B", "C", "A"],
            "snapshot_date": [
                "2024-01-30",
                "2024-01-30",
                "2024-01-31",
                "2024-01-31",
                "2024-01-31",
                "2024-02-29",
            ],
        }
    )

    selected = _monthly_rebalance_snapshots(frame)

    assert sorted(pd.to_datetime(selected["snapshot_date"]).dt.date.unique()) == [
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-02-29").date(),
    ]


def test_ashare_financial_parser_preserves_per_share_history() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _ashare_financial_records_from_frame,
    )

    records = _ashare_financial_records_from_frame(
        "000001",
        pd.DataFrame(
            {
                "报告期": ["2024-12-31"],
                "基本每股收益": ["2.00"],
                "每股净资产": ["10.00"],
                "每股经营现金流": ["1.20"],
                "净资产收益率": ["20%"],
                "资产负债率": ["60%"],
            }
        ),
    )

    assert records[0]["report_date"] == pd.Timestamp("2025-04-30").date()
    assert records[0]["eps"] == pytest.approx(2.0)
    assert records[0]["book_value_per_share"] == pytest.approx(10.0)
    assert records[0]["operating_cash_flow"] is None
    assert records[0]["operating_cash_flow_per_share"] == pytest.approx(1.2)


def test_ashare_bulk_financial_parser_normalizes_codes_and_percentages() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _ashare_bulk_financial_records_from_frame,
    )

    records = _ashare_bulk_financial_records_from_frame(
        pd.Timestamp("2024-09-30").date(),
        pd.DataFrame(
            {
                "股票代码": ["1.0", "invalid"],
                "营业总收入-营业总收入": [1_000_000.0, 2_000_000.0],
                "净利润-净利润": [100_000.0, 200_000.0],
                "销售毛利率": ["35.5%", "20%"],
                "净资产收益率": ["12.5%", "10%"],
                "每股收益": ["0.50", "1.00"],
                "每股净资产": ["4.00", "5.00"],
                "每股经营现金流量": ["0.75", "0.80"],
            }
        ),
    )

    assert len(records) == 1
    assert records[0]["ticker"] == "000001"
    assert records[0]["report_date"] == pd.Timestamp("2024-11-29").date()
    assert records[0]["gross_margin"] == pytest.approx(0.355)
    assert records[0]["roe"] == pytest.approx(0.125)
    assert records[0]["eps"] == pytest.approx(0.5)
    assert records[0]["book_value_per_share"] == pytest.approx(4.0)
    assert records[0]["operating_cash_flow_per_share"] == pytest.approx(0.75)


def test_hk_financial_parser_preserves_currency_and_statement_duration() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _hk_financial_records_from_statements,
    )

    period = pd.Timestamp("2024-12-31")
    income = pd.DataFrame(
        {period: [1_000.0, 100.0, 0.50]},
        index=["Total Revenue", "Net Income", "Basic EPS"],
    )
    balance = pd.DataFrame(
        {period: [2_000.0, 500.0, 200.0]},
        index=["Total Assets", "Stockholders Equity", "Ordinary Shares Number"],
    )
    cashflow = pd.DataFrame(
        {period: [120.0]},
        index=["Operating Cash Flow"],
    )

    records = _hk_financial_records_from_statements(
        "0700.HK",
        income,
        balance,
        cashflow,
        statement_months=12,
        currency_to_rmb=1.0,
    )

    assert records[0]["report_date"] == pd.Timestamp("2025-04-30").date()
    assert records[0]["eps"] == pytest.approx(0.5)
    assert records[0]["book_value_per_share"] == pytest.approx(2.5)
    assert records[0]["operating_cash_flow_per_share"] == pytest.approx(0.6)
    assert records[0]["statement_months"] == 12
    assert records[0]["currency_to_rmb"] == pytest.approx(1.0)


def test_ashare_statement_parser_merges_quality_and_cashflow_features() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _ashare_statement_financial_records_from_frames,
    )

    balance = pd.DataFrame(
        [
            {
                "股票代码": "000001",
                "资产-总资产": 1_000.0,
                "负债-总负债": 400.0,
                "股东权益合计": 600.0,
                "资产负债率": 40.0,
            }
        ]
    )
    cashflow = pd.DataFrame(
        [
            {
                "股票代码": "000001",
                "经营性现金流-现金流量净额": 80.0,
            }
        ]
    )
    income = pd.DataFrame(
        [
            {
                "股票代码": "000001",
                "营业总收入": 500.0,
                "净利润": 60.0,
                "营业总支出-营业支出": 300.0,
            }
        ]
    )

    records = _ashare_statement_financial_records_from_frames(
        pd.Timestamp("2024-06-30").date(),
        balance=balance,
        cashflow=cashflow,
        income=income,
    )

    assert len(records) == 1
    record = records[0]
    assert record["report_date"] == pd.Timestamp("2024-09-28").date()
    assert record["statement_months"] == pytest.approx(6.0)
    assert record["total_assets"] == pytest.approx(1_000.0)
    assert record["total_equity"] == pytest.approx(600.0)
    assert record["operating_cash_flow"] == pytest.approx(80.0)
    assert record["gross_margin"] == pytest.approx(0.4)
    assert record["net_margin"] == pytest.approx(0.12)
    assert record["roe"] == pytest.approx(0.2)
    assert record["roa"] == pytest.approx(0.12)
    assert record["debt_to_equity"] == pytest.approx(0.4)


def test_financial_flow_features_are_annualized_from_statement_duration() -> None:
    from valueinvestor.scorer_improver.ground_truth import (
        annualize_financial_feature_frame,
    )

    frame = pd.DataFrame(
        {
            "statement_months": [3.0, 6.0, 12.0, None],
            "revenue": [10.0, 20.0, 40.0, 50.0],
            "net_income": [1.0, 2.0, 4.0, 5.0],
            "total_assets": [100.0, 100.0, 100.0, 100.0],
        }
    )

    result = annualize_financial_feature_frame(frame)

    assert result["revenue"].tolist() == pytest.approx([40.0, 40.0, 40.0, 50.0])
    assert result["net_income"].tolist() == pytest.approx([4.0, 4.0, 4.0, 5.0])
    assert result["total_assets"].tolist() == pytest.approx([100.0] * 4)


def test_financial_history_matches_live_annual_period_policy() -> None:
    from valueinvestor.scorer_improver.ground_truth import (
        align_financial_history_to_live_periods,
    )

    frame = pd.DataFrame(
        {
            "ticker": ["NEW", "NEW", "NEW", "FISCAL.HK", "FISCAL.HK", "NOANNUAL"],
            "period": [
                "2023-06-30",
                "2023-12-31",
                "2024-03-31",
                "2023-06-30",
                "2023-09-30",
                "2024-06-30",
            ],
            "report_date": [
                "2023-08-30",
                "2024-03-30",
                "2024-04-30",
                "2023-09-30",
                "2023-12-30",
                "2024-08-30",
            ],
            "statement_months": [6.0, 12.0, 3.0, 12.0, 3.0, 6.0],
        }
    )

    result = align_financial_history_to_live_periods(frame)

    assert list(result.loc[result["ticker"] == "NEW", "period"]) == [
        "2023-06-30",
        "2023-12-31",
    ]
    assert list(result.loc[result["ticker"] == "FISCAL.HK", "period"]) == ["2023-06-30"]
    assert list(result.loc[result["ticker"] == "NOANNUAL", "period"]) == ["2024-06-30"]


def test_valuation_merge_preserves_complementary_fields() -> None:
    from valueinvestor.scorer_improver.data_prep import _ordered_valuation_frame

    merged = _ordered_valuation_frame(
        [
            {"ticker": "AAA", "date": "2024-01-31", "price": 10.0, "market_cap_rmb": 9e9},
            {"ticker": "AAA", "date": "2024-01-31", "pe_ratio": 8.0, "pb_ratio": 1.2},
        ]
    )

    assert len(merged) == 1
    assert merged.loc[0, "price"] == pytest.approx(10.0)
    assert merged.loc[0, "market_cap_rmb"] == pytest.approx(9e9)
    assert merged.loc[0, "pe_ratio"] == pytest.approx(8.0)
    assert merged.loc[0, "pb_ratio"] == pytest.approx(1.2)


def test_derived_valuation_ratios_obey_report_availability() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _derive_historical_valuation_ratios,
        _ordered_financial_frame,
    )

    financials = _ordered_financial_frame(
        [
            {
                "ticker": "AAA",
                "period": "2023-12-31",
                "report_date": "2024-04-29",
                "eps": 2.0,
                "book_value_per_share": 8.0,
                "revenue": 100.0,
            },
        ]
    )
    valuations = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-04-01", "price": 20.0, "market_cap_rmb": 1000.0},
            {"ticker": "AAA", "date": "2024-05-01", "price": 20.0, "market_cap_rmb": 1000.0},
        ]
    )

    derived = _derive_historical_valuation_ratios(valuations, financials)

    assert pd.isna(derived.loc[0, "pe_ratio"])
    assert derived.loc[1, "pe_ratio"] == pytest.approx(10.0)
    assert derived.loc[1, "pb_ratio"] == pytest.approx(2.5)
    assert derived.loc[1, "ps_ratio"] == pytest.approx(10.0)


def test_hk_valuation_ratios_use_currency_adjusted_statements() -> None:
    from valueinvestor.scorer_improver.data_prep import (
        _derive_historical_valuation_ratios,
        _ordered_financial_frame,
    )

    financials = _ordered_financial_frame(
        [
            {
                "ticker": "0005.HK",
                "period": "2024-12-31",
                "report_date": "2025-04-30",
                "net_income": 10.0,
                "total_equity": 100.0,
                "revenue": 200.0,
                "statement_months": 12,
                "currency_to_rmb": 7.2,
            },
        ]
    )
    valuations = pd.DataFrame(
        [
            {
                "ticker": "0005.HK",
                "date": "2025-05-01",
                "price": 10.0,
                "market_cap_rmb": 720.0,
            },
        ]
    )

    derived = _derive_historical_valuation_ratios(valuations, financials)

    assert derived.loc[0, "pe_ratio"] == pytest.approx(10.0)
    assert derived.loc[0, "pb_ratio"] == pytest.approx(1.0)
    assert derived.loc[0, "ps_ratio"] == pytest.approx(0.5)


def test_promotion_gate_rejects_negative_latest_year_rho() -> None:
    from valueinvestor.scorer_improver.promotion_gate import evaluate_promotion_gate

    def metrics(rho: float) -> dict[str, dict[str, float]]:
        return {
            horizon: {
                "spearman_rho": rho,
                "hit_rate_top20": 0.6,
                "mean_excess_return": 0.1,
            }
            for horizon in ("1m", "3m", "6m")
        }

    result = evaluate_promotion_gate(
        candidate_metrics=metrics(0.2),
        incumbent_metrics=metrics(0.1),
        manifest={"gate_snapshots": 24},
        regime_diagnostics={
            "year": [
                {"label": "2024", "candidate_primary": 0.3, "deltas": {"6m": 0.1}},
                {"label": "2025", "candidate_primary": -0.01, "deltas": {"6m": 0.01}},
            ],
        },
    )

    assert not result.accepted
    assert result.reason == "recent-year 6m rho below gate"


def test_strict_outer_gate_records_exactly_one_candidate_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from valueinvestor.scorer_improver import ml_trainer
    from valueinvestor.scorer_improver.promotion_gate import HoldoutGateConfig
    from valueinvestor.screener.ml_ranker import MODEL_SCHEMA_VERSION, expected_feature_names

    rows = []
    for date_index, snapshot_date in enumerate(pd.date_range("2015-01-31", periods=120, freq="ME").date):
        for ticker_index in range(20):
            rank = ticker_index / 19.0
            rows.append(
                {
                    "ticker": f"T{ticker_index:03d}",
                    "snapshot_date": snapshot_date,
                    "close": 10.0 + rank,
                    "market_cap_rmb": 10e9 + ticker_index,
                    "pe_ratio": 5.0 + rank,
                    "pb_ratio": 1.0 + rank,
                    "roe": 0.10 + rank / 10.0,
                    "debt_to_equity": 0.2,
                    "forward_return_1w": rank + date_index * 1e-5,
                    "forward_return_1m": rank + date_index * 1e-5,
                    "forward_return_3m": rank + date_index * 1e-5,
                    "forward_return_6m": rank + date_index * 1e-5,
                }
            )
    snapshots = pd.DataFrame(rows)
    output_model_path = tmp_path / "model.json"
    feature_names = expected_feature_names()
    output_model_path.write_text(
        json.dumps({
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": [0.0] * len(feature_names),
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {"target_horizon": "6m"},
        }),
        encoding="utf-8",
    )
    ledger_records = []
    walk_forward_references = []
    original_walk_forward = ml_trainer._walk_forward_validate_candidate

    def capture_walk_forward(*args, **kwargs):
        walk_forward_references.append(kwargs.get("incumbent_payload"))
        return original_walk_forward(*args, **kwargs)

    monkeypatch.setattr(
        ml_trainer,
        "_walk_forward_validate_candidate",
        capture_walk_forward,
    )
    monkeypatch.setattr(
        ml_trainer,
        "prepare_ml_training_snapshots",
        lambda **kwargs: snapshots,
    )
    monkeypatch.setattr(
        ml_trainer,
        "append_gate_ledger",
        lambda result, **kwargs: ledger_records.append((result, kwargs)),
    )

    metadata = ml_trainer.train_ml_ranker(
        output_model_path=output_model_path,
        backend="numpy",
        model_kind="ridge-only",
        snapshot_frequency="quarterly",
        target_improvement=-999.0,
        max_training_rows=0,
        candidate_feature_sets=("core",),
        candidate_prior_strategies=("no-ticker-priors",),
        candidate_targets=("target-rank-6m",),
        candidate_ridge_lambdas=("10",),
        walk_forward_folds=1,
        walk_forward_validation_months=24,
        walk_forward_min_6m_delta=-999.0,
        walk_forward_max_horizon_degradation=999.0,
        gate_config=HoldoutGateConfig(
            holdout_months=12,
            embargo_days=197,
            min_6m_delta=-999.0,
            max_horizon_degradation=999.0,
            min_weighted_utility=-999.0,
            min_gate_snapshots=4,
            min_train_snapshots=60,
            require_6m_top20_excess_non_degradation=False,
        ),
        strict_outer_gate=True,
    )

    assert metadata["strict_outer_gate"] is True
    assert walk_forward_references == [None]
    assert metadata["walk_forward"]["reference"] == "hand_scorer"
    assert len(metadata["promotion_gate_attempts"]) == 1
    assert len(ledger_records) == 1


# ── ExperimentLog tests ─────────────────────────────────────────────────────


class TestExperimentLog:
    """Tests for ExperimentLog JSONL logger."""

    def test_log_creates_file(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.25, baseline_rho=0.20, kept=True, description="test")
        assert log_file.exists()
        records = log.read_all()
        assert len(records) == 1
        assert records[0]["iteration"] == 1
        assert records[0]["spearman_rho"] == 0.25
        assert records[0]["kept"] is True
        assert records[0]["delta"] == 0.05

    def test_log_preserves_zero_6m_fields(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.0, baseline_rho=0.0, kept=False, description="zero")
        record = log.read_all()[0]
        assert record["spearman_rho_6m"] == 0.0
        assert record["baseline_rho_6m"] == 0.0

    def test_read_last_n(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        for i in range(20):
            log.log(iteration=i, spearman_rho=0.1 * i, baseline_rho=0.0, kept=True, description=f"iter-{i}")
        assert log.total_experiments() == 20
        last5 = log.read_last_n(5)
        assert len(last5) == 5
        assert last5[0]["iteration"] == 15
        assert last5[-1]["iteration"] == 19

    def test_best_rho(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.1, baseline_rho=0.0, kept=True, description="a")
        log.log(iteration=2, spearman_rho=0.5, baseline_rho=0.1, kept=True, description="b")
        log.log(iteration=3, spearman_rho=0.3, baseline_rho=0.5, kept=False, description="c")
        assert log.best_rho() == 0.5

    def test_best_rho_no_kept(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.1, baseline_rho=0.2, kept=False, description="x")
        assert log.best_rho() is None

    def test_current_lineage_ignores_older_branch(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.10, baseline_rho=0.00, kept=True, description="a")
        log.log(iteration=2, spearman_rho=0.50, baseline_rho=0.10, kept=True, description="b")
        log.log(iteration=3, spearman_rho=0.20, baseline_rho=0.20, kept=True, description="manual reset")
        log.log(iteration=4, spearman_rho=0.18, baseline_rho=0.20, kept=False, description="reverted")
        log.log(iteration=5, spearman_rho=0.30, baseline_rho=0.20, kept=True, description="improved")

        current = log.read_current_lineage()
        assert [record["iteration"] for record in current] == [3, 4, 5]
        assert log.read_last_n(2, current_lineage=True)[0]["iteration"] == 4
        assert log.best_rho(current_lineage=True) == 0.30
        assert log.best_rho() == 0.50

    def test_best_rho_filters_by_ground_truth_id(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import experiment_log
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        gt_id = "ground_truth.parquet:1:100"
        monkeypatch.setattr(
            "valueinvestor.scorer_improver.ground_truth.ground_truth_fingerprint",
            lambda: gt_id,
        )

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        log.log(iteration=1, spearman_rho=0.20, baseline_rho=0.10, kept=True, description="current")

        old_record = log.read_all()[0] | {
            "iteration": 2,
            "spearman_rho": 0.50,
            "spearman_rho_1m": 0.40,
            "spearman_rho_3m": 0.45,
            "ground_truth_id": "old-ground-truth",
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(experiment_log.json.dumps(old_record) + "\n")

        assert log.best_rho(ground_truth_id=gt_id) == 0.20
        assert log.best_rho(ground_truth_id="old-ground-truth") == 0.50
        assert log.best_rho_per_horizon(ground_truth_id=gt_id)["6m"] == 0.20

    def test_empty_log(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        log_file = tmp_path / "exp.jsonl"
        log = ExperimentLog(path=log_file)
        assert log.total_experiments() == 0
        assert log.read_last_n(5) == []
        assert log.best_rho() is None
        assert log.read_all() == []


# ── Agent helper tests ──────────────────────────────────────────────────────


class TestAgentHelpers:
    """Tests for agent.py helper functions (code extraction, diff, etc.)."""

    def test_ml_ranker_feature_schema_vector_length(self) -> None:
        from valueinvestor.screener.ml_ranker import (
            expected_feature_names,
            feature_vector_from_values,
        )

        values = {
            "close": 10.0,
            "pe_ratio": 8.0,
            "pb_ratio": 1.2,
            "market_cap_rmb": 10_000_000_000.0,
            "net_income": 100_000_000.0,
            "total_assets": 1_000_000_000.0,
            "gross_margin": 0.3,
            "roe": 0.15,
            "composite_score": 60.0,
            "value_score": 65.0,
            "quality_score": 55.0,
            "growth_score": 50.0,
        }

        vector = feature_vector_from_values(values, "TEST", {})

        assert len(vector) == len(expected_feature_names())

    def test_ml_ranker_short_horizon_feature_schema_vector_length(self) -> None:
        from valueinvestor.screener.ml_ranker import (
            expected_feature_names,
            feature_vector_from_values,
        )

        feature_names = expected_feature_names(include_short_horizon=True)
        values = {
            "close": 10.0,
            "composite_score": 60.0,
            "value_score": 65.0,
            "quality_score": 55.0,
            "growth_score": 50.0,
            "price_return_5d": 0.03,
            "relative_return_5d": 0.01,
            "price_volatility_21d": 0.02,
        }

        vector = feature_vector_from_values(
            values,
            "TEST",
            {},
            feature_names=feature_names,
        )

        assert len(vector) == len(feature_names)
        assert len(feature_names) > len(expected_feature_names())
        assert vector[feature_names.index("price_return_5d")] == pytest.approx(0.03)
        assert vector[feature_names.index("market_return_5d_miss")] == 1.0

    def test_ml_ranker_expanded_feature_schema_includes_interactions_and_polynomial_terms(self) -> None:
        from valueinvestor.screener.ml_ranker import (
            expected_feature_names,
            feature_vector_from_values,
        )

        feature_names = expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
        )
        vector = feature_vector_from_values(
            {
                "value_score": 2.0,
                "quality_score": 3.0,
                "growth_score": 5.0,
                "roe": 0.2,
                "net_income": 10.0,
                "market_cap_rmb": 100.0,
                "free_cash_flow": 4.0,
                "relative_return_63d": 0.7,
            },
            "TEST",
            {"TEST": {"ticker_rankmean_6m": 0.4}},
            feature_names=feature_names,
        )

        assert "value_quality_score" in feature_names
        assert vector[feature_names.index("value_quality_score")] == pytest.approx(6.0)
        assert vector[
            feature_names.index("ticker_rankmean_6m_relative_return_63d")
        ] == pytest.approx(0.28)
        assert "value_score_sq" in feature_names
        assert vector[feature_names.index("value_score_sq")] == pytest.approx(4.0)
        assert vector[feature_names.index("ticker_rankmean_6m_sq")] == pytest.approx(0.16)

    def test_ml_ranker_loads_expanded_and_poly_artifact_schemas(self, tmp_path: Path) -> None:
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            load_model,
            stable_factor_feature_names,
        )

        schemas = (
            stable_factor_feature_names(),
            expected_feature_names(
                include_short_horizon=True,
                include_interactions=True,
            ),
            expected_feature_names(
                include_short_horizon=True,
                include_interactions=True,
                include_polynomial=True,
            ),
        )
        for index, feature_names in enumerate(schemas):
            model_path = tmp_path / f"model_{index}.json"
            model_path.write_text(
                json.dumps({
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "feature_names": feature_names,
                    "clip_low": [-100.0] * len(feature_names),
                    "clip_high": [100.0] * len(feature_names),
                    "mean": [0.0] * len(feature_names),
                    "scale": [1.0] * len(feature_names),
                    "coef": [0.0] * len(feature_names),
                    "intercept": 0.0,
                    "ticker_priors": {},
                    "metadata": {},
                }),
                encoding="utf-8",
            )

            clear_model_cache()
            model = load_model(model_path)

            assert model is not None
            assert len(model.feature_names) == len(feature_names)

        clear_model_cache()

    def test_ml_ranker_cross_sectional_schema_scores_with_snapshot_ranks(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
            include_cross_sectional=True,
        )
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("cs_rank_value_score")] = 1.0
        model_path = tmp_path / "model_cs.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": coef,
                "intercept": 0.0,
                "ticker_priors": {},
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="LOW", name="L", market=Market.A_SHARE),
                financials=Financials(ticker="LOW", period="snapshot"),
                valuation=ValuationMetrics(ticker="LOW", date="snapshot", price=1.0),
                value_score=10.0,
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="HIGH", name="H", market=Market.A_SHARE),
                financials=Financials(ticker="HIGH", period="snapshot"),
                valuation=ValuationMetrics(ticker="HIGH", date="snapshot", price=1.0),
                value_score=90.0,
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path)

        assert results[1].composite_score > results[0].composite_score
        clear_model_cache()

    def test_ml_ranker_target_rank_1w_market_ranks_within_market(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _target_rank_by_snapshot_market

        df = pd.DataFrame({
            "snapshot_date": [pd.Timestamp("2024-01-01")] * 4,
            "ticker": ["AAA", "BBB", "0700.HK", "0005.HK"],
            "forward_return_1w": [0.10, 0.20, -0.10, 0.50],
        })

        target = _target_rank_by_snapshot_market(df, "forward_return_1w")

        assert target.iloc[0] == pytest.approx(-1.0 / 3.0)
        assert target.iloc[1] == pytest.approx(1.0 / 3.0)
        assert target.iloc[2] == pytest.approx(-1.0 / 3.0)
        assert target.iloc[3] == pytest.approx(1.0 / 3.0)

    def test_ml_ranker_training_cross_sectional_features_rank_by_snapshot_and_market(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import build_feature_matrix

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB", "0700.HK", "AAA"],
            "snapshot_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "value_score": [10.0, 20.0, 30.0, 40.0],
        })

        matrix = build_feature_matrix(
            frame,
            {},
            feature_names=[
                "cs_rank_value_score",
                "market_cs_rank_value_score",
            ],
        )

        assert matrix[:, 0].tolist() == pytest.approx([-0.5, 0.0, 0.5, 0.0])
        assert matrix[:, 1].tolist() == pytest.approx([-0.5, 0.5, 0.0, 0.0])

    def test_ml_ranker_cross_sectional_ties_are_order_invariant(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import build_feature_matrix

        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "snapshot_date": ["2024-01-01"] * 3,
                "value_score": [10.0, 10.0, 20.0],
            }
        )
        feature_names = ["cs_rank_value_score"]

        original = build_feature_matrix(
            frame,
            {},
            feature_names=feature_names,
        )[:, 0]
        shuffled_frame = frame.iloc[[2, 0, 1]].reset_index(drop=True)
        shuffled = build_feature_matrix(
            shuffled_frame,
            {},
            feature_names=feature_names,
        )[:, 0]
        shuffled_by_ticker = dict(zip(shuffled_frame["ticker"], shuffled))

        assert original.tolist() == pytest.approx([-0.25, -0.25, 0.5])
        assert [shuffled_by_ticker[ticker] for ticker in frame["ticker"]] == pytest.approx(original)

    def test_ml_ranker_application_blend_matches_offline_and_runtime(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("value_score")] = 1.0
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coef,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {
                "application_blend_candidate_weight": 0.25,
                "application_blend_scale": "snapshot_rank",
            },
        }
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "snapshot_date": ["2025-01-02"] * 5,
                "value_score": [10.0, 30.0, 50.0, 70.0, 90.0],
                "composite_score": [90.0, 70.0, 50.0, 30.0, 10.0],
            }
        )

        offline = predict_ml_ranker_payload(frame, payload)
        assert offline.tolist() == pytest.approx([1.0 / 3.0, 1.0 / 6.0, 0.0, -1.0 / 6.0, -1.0 / 3.0])

        model_path = tmp_path / "application_blend.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        results = [
            ScreeningResult(
                company=Company(ticker=ticker, name=ticker, market=Market.A_SHARE),
                financials=Financials(ticker=ticker, period="snapshot"),
                valuation=ValuationMetrics(
                    ticker=ticker,
                    date="2025-01-02",
                    price=1.0,
                ),
                value_score=value_score,
                composite_score=composite_score,
            )
            for ticker, value_score, composite_score in zip(
                frame["ticker"],
                frame["value_score"],
                frame["composite_score"],
            )
        ]
        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(offline)
        assert results[0].composite_score > results[4].composite_score

    def test_ml_ranker_application_factor_blend_matches_offline_and_runtime(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload
        from valueinvestor.screener import ml_ranker

        feature_names = ml_ranker.expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("value_score")] = 1.0
        components = [
            {"signal": "model", "weight": 0.10},
            {"signal": "quality_de_crowding", "weight": 0.135},
            {"signal": "gross_profitability_de_crowding", "weight": 0.12},
            {"signal": "roe_de_crowding", "weight": 0.08},
            {"signal": "roa_de_crowding", "weight": 0.07},
            {"signal": "book_yield", "weight": 0.18},
            {"signal": "sales_yield", "weight": 0.045},
            {"signal": "liability_yield", "weight": 0.135},
            {"signal": "momentum_63d", "weight": 0.27},
            {"signal": "low_current_ratio", "weight": 0.135},
        ]
        payload = {
            "schema_version": ml_ranker.MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coef,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {
                "application_factor_template": "value_momentum",
                "application_factor_components": components,
                "application_factor_scale": "snapshot_rank",
            },
        }
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "snapshot_date": ["2025-01-02"] * 5,
                "value_score": [10.0, 20.0, 30.0, 40.0, 50.0],
                "quality_score": [50.0, 40.0, 30.0, 20.0, 10.0],
                "revenue": [100.0, 200.0, 150.0, 250.0, 300.0],
                "gross_margin": [0.10, 0.25, 0.20, 0.30, 0.15],
                "total_assets": [200.0, 250.0, 300.0, 350.0, 400.0],
                "roe": [0.20, 0.15, 0.10, 0.05, -0.05],
                "roa": [0.08, 0.06, 0.04, 0.02, -0.01],
                "pb_ratio": [5.0, 4.0, 3.0, 2.0, 1.0],
                "ps_ratio": [5.0, 1.0, 4.0, 2.0, 3.0],
                "total_liabilities": [10.0, 20.0, 30.0, 40.0, 50.0],
                "market_cap_rmb": [100.0] * 5,
                "relative_return_63d": [-0.2, -0.1, 0.0, 0.1, 0.2],
                "current_ratio": [5.0, None, 3.0, 2.0, 1.0],
            }
        )

        offline = predict_ml_ranker_payload(frame, payload)
        assert np.isfinite(offline).all()

        model_path = tmp_path / "application_factor_blend.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        results = [
            ScreeningResult(
                company=Company(ticker=row.ticker, name=row.ticker, market=Market.A_SHARE),
                financials=Financials(
                    ticker=row.ticker,
                    period="snapshot",
                    revenue=row.revenue,
                    gross_margin=row.gross_margin,
                    total_assets=row.total_assets,
                    total_liabilities=row.total_liabilities,
                    roe=row.roe,
                    roa=row.roa,
                    current_ratio=row.current_ratio,
                ),
                valuation=ValuationMetrics(
                    ticker=row.ticker,
                    date="2025-01-02",
                    price=1.0,
                    pb_ratio=row.pb_ratio,
                    ps_ratio=row.ps_ratio,
                    market_cap_rmb=row.market_cap_rmb,
                ),
                value_score=row.value_score,
                quality_score=row.quality_score,
                composite_score=50.0,
            )
            for row in frame.itertuples(index=False)
        ]
        momentum = frame["relative_return_63d"].tolist()
        monkeypatch.setattr(
            ml_ranker,
            "_recent_price_features_for_results",
            lambda _results: {
                index: {"relative_return_63d": value}
                for index, value in enumerate(momentum)
            },
        )
        ml_ranker.clear_model_cache()
        assert ml_ranker.score_results_with_ml_ranker(results, model_path=model_path)
        ml_ranker.clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(offline)

    def test_ml_ranker_quality_residual_matches_offline_and_runtime(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("value_score")] = 1.0
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coef,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {
                "application_quality_residual_candidate_weight": 0.5,
                "application_quality_residual_direction": "inverse",
                "application_quality_residual_scale": "snapshot_rank",
            },
        }
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "snapshot_date": ["2025-01-02"] * 5,
                "value_score": [10.0, 30.0, 50.0, 70.0, 90.0],
                "quality_score": [10.0, 90.0, 50.0, 30.0, 70.0],
            }
        )

        offline = predict_ml_ranker_payload(frame, payload)
        assert offline.tolist() == pytest.approx([0.0, -0.5, 0.0, 1.0 / 3.0, 1.0 / 6.0])

        model_path = tmp_path / "quality_residual.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        results = [
            ScreeningResult(
                company=Company(ticker=ticker, name=ticker, market=Market.A_SHARE),
                financials=Financials(ticker=ticker, period="snapshot"),
                valuation=ValuationMetrics(
                    ticker=ticker,
                    date="2025-01-02",
                    price=1.0,
                ),
                value_score=value_score,
                quality_score=quality_score,
                composite_score=50.0,
            )
            for ticker, value_score, quality_score in zip(
                frame["ticker"],
                frame["value_score"],
                frame["quality_score"],
            )
        ]
        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(offline)
        assert results[3].composite_score > results[4].composite_score
        assert results[4].composite_score > results[1].composite_score

        frame["roe"] = [0.20, 0.15, 0.10, 0.05, -0.05]
        payload["metadata"] = {
            "application_residual_candidate_weight": 0.5,
            "application_residual_signal": "roe_de_crowding",
            "application_residual_column": "roe",
            "application_residual_direction": -1.0,
            "application_residual_scale": "snapshot_rank",
        }
        roe_offline = predict_ml_ranker_payload(frame, payload)
        roe_model_path = tmp_path / "roe_residual.json"
        roe_model_path.write_text(json.dumps(payload), encoding="utf-8")
        for result, roe in zip(results, frame["roe"]):
            result.financials.roe = roe
        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=roe_model_path)
        clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(roe_offline)

    def test_ml_ranker_gross_profitability_de_crowding_matches_runtime(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.ml_trainer import (
            _prediction_rank_by_snapshot,
            predict_ml_ranker_payload,
        )
        from valueinvestor.screener import ml_ranker

        feature_names = ml_ranker.expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("value_score")] = 1.0
        metadata = {
            "application_gross_profitability_de_crowding_factor_weight": 0.16,
            "application_gross_profitability_de_crowding_max_model_rank": 0.4,
            "application_gross_profitability_de_crowding_min_market_return_63d": 0.0,
        }
        payload = {
            "schema_version": ml_ranker.MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coef,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": metadata,
        }
        tickers = [f"T{index:02d}" for index in range(10)]
        value_scores = [float(index * 10) for index in range(1, 11)]
        gross_margins = [float(index) / 20.0 for index in range(1, 11)]
        frame = pd.DataFrame({
            "ticker": tickers,
            "snapshot_date": ["2025-01-02"] * len(tickers),
            "value_score": value_scores,
            "revenue": [100.0] * len(tickers),
            "gross_margin": gross_margins,
            "total_assets": [100.0] * len(tickers),
            "market_return_63d": [0.05] * len(tickers),
        })

        base_payload = {**payload, "metadata": {}}
        base_predictions = predict_ml_ranker_payload(frame, base_payload)
        base_rank = _prediction_rank_by_snapshot(frame, base_predictions)
        offline = predict_ml_ranker_payload(frame, payload)
        protected = base_rank > 0.4

        assert offline[protected].tolist() == pytest.approx(base_rank[protected])
        assert not np.allclose(offline[~protected], base_rank[~protected])
        inactive_frame = frame.assign(market_return_63d=-0.05)
        assert predict_ml_ranker_payload(inactive_frame, payload).tolist() == pytest.approx(
            base_rank
        )

        model_path = tmp_path / "gross_profitability_de_crowding.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        results = [
            ScreeningResult(
                company=Company(ticker=ticker, name=ticker, market=Market.A_SHARE),
                financials=Financials(
                    ticker=ticker,
                    period="snapshot",
                    revenue=100.0,
                    gross_margin=gross_margin,
                    total_assets=100.0,
                ),
                valuation=ValuationMetrics(
                    ticker=ticker,
                    date="2025-01-02",
                    price=1.0,
                ),
                value_score=value_score,
                composite_score=50.0,
            )
            for ticker, value_score, gross_margin in zip(
                tickers,
                value_scores,
                gross_margins,
            )
        ]
        monkeypatch.setattr(
            ml_ranker,
            "_recent_price_features_for_results",
            lambda _results: {
                index: {"market_return_63d": 0.05}
                for index in range(len(_results))
            },
        )

        ml_ranker.clear_model_cache()
        assert ml_ranker.score_results_with_ml_ranker(results, model_path=model_path)
        ml_ranker.clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(offline)

    def test_ml_ranker_book_yield_residual_matches_offline_and_runtime(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("value_score")] = 1.0
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": coef,
            "intercept": 0.0,
            "ticker_priors": {},
            "metadata": {
                "application_residual_candidate_weight": 0.7,
                "application_residual_signal": "book_yield",
                "application_residual_column": "pb_ratio",
                "application_residual_direction": -1.0,
                "application_residual_scale": "snapshot_rank",
            },
        }
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "snapshot_date": ["2025-01-02"] * 5,
                "value_score": [10.0, 30.0, 50.0, 70.0, 90.0],
                "pb_ratio": [5.0, 1.0, 3.0, 4.0, 2.0],
            }
        )

        offline = predict_ml_ranker_payload(frame, payload)
        assert offline.tolist() == pytest.approx(
            [
                -2.0 / 3.0,
                -1.0 / 30.0,
                0.0,
                2.0 / 15.0,
                17.0 / 30.0,
            ]
        )

        model_path = tmp_path / "book_yield_residual.json"
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        results = [
            ScreeningResult(
                company=Company(ticker=ticker, name=ticker, market=Market.A_SHARE),
                financials=Financials(ticker=ticker, period="snapshot"),
                valuation=ValuationMetrics(
                    ticker=ticker,
                    date="2025-01-02",
                    price=1.0,
                    pb_ratio=pb_ratio,
                ),
                value_score=value_score,
                composite_score=50.0,
            )
            for ticker, value_score, pb_ratio in zip(
                frame["ticker"],
                frame["value_score"],
                frame["pb_ratio"],
            )
        ]
        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert [result._ml_ranker_raw_score for result in results] == pytest.approx(offline)

    def test_ml_ranker_cross_sectional_prior_interaction_features(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import build_feature_matrix
        from valueinvestor.screener.ml_ranker import expected_feature_names

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "snapshot_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "value_score": [10.0, 20.0, 30.0],
        })
        feature_names = [
            "cs_rank_value_score",
            "cs_rank_ticker_rankmean_6m",
            "csx_value_ticker_rankmean_6m",
        ]

        matrix = build_feature_matrix(
            frame,
            {
                "AAA": {"ticker_rankmean_6m": 0.0},
                "BBB": {"ticker_rankmean_6m": 1.0},
                "CCC": {"ticker_rankmean_6m": 2.0},
            },
            feature_names=feature_names,
        )

        assert "csx_value_ticker_rankmean_6m" in expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
            include_cross_sectional=True,
            include_cross_sectional_interactions=True,
        )
        assert matrix[:, 0].tolist() == pytest.approx([-0.5, 0.0, 0.5])
        assert matrix[:, 1].tolist() == pytest.approx([-0.5, 0.0, 0.5])
        assert matrix[:, 2].tolist() == pytest.approx([0.25, 0.0, 0.25])

    def test_ml_ranker_scores_cross_sectional_interaction_artifact(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names(
            include_short_horizon=True,
            include_interactions=True,
            include_polynomial=True,
            include_cross_sectional=True,
            include_cross_sectional_interactions=True,
        )
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("csx_value_ticker_rankmean_6m")] = 1.0
        model_path = tmp_path / "model_csx.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": coef,
                "intercept": 0.0,
                "ticker_priors": {
                    "LOW": {"ticker_rankmean_6m": 0.0},
                    "MID": {"ticker_rankmean_6m": 1.0},
                    "HIGH": {"ticker_rankmean_6m": 2.0},
                },
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="LOW", name="L", market=Market.A_SHARE),
                financials=Financials(ticker="LOW", period="snapshot"),
                valuation=ValuationMetrics(ticker="LOW", date="snapshot", price=1.0),
                value_score=10.0,
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="MID", name="M", market=Market.A_SHARE),
                financials=Financials(ticker="MID", period="snapshot"),
                valuation=ValuationMetrics(ticker="MID", date="snapshot", price=1.0),
                value_score=20.0,
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="HIGH", name="H", market=Market.A_SHARE),
                financials=Financials(ticker="HIGH", period="snapshot"),
                valuation=ValuationMetrics(ticker="HIGH", date="snapshot", price=1.0),
                value_score=30.0,
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path)

        assert results[0].composite_score > results[1].composite_score
        assert results[2].composite_score > results[1].composite_score
        clear_model_cache()

    def test_recent_positive_year_start_selects_latest_improving_gate_year(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _recent_positive_year_start

        diagnostics = {
            "year": [
                {"label": "2023", "deltas": {"6m": -0.2}},
                {"label": "2024", "deltas": {"6m": -0.1}},
                {"label": "2025", "deltas": {"6m": 0.04}},
            ]
        }

        assert _recent_positive_year_start(
            diagnostics,
            gate_start="2023-07-07",
            primary_horizon="6m",
        ) == "2025-01-01"

    def test_recent_positive_year_market_route_selects_only_improving_markets(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _recent_positive_year_market_route,
        )

        rows = []
        candidate_predictions = []
        incumbent_predictions = []
        for snapshot_idx, snapshot_date in enumerate(pd.date_range("2025-01-01", periods=4, freq="MS")):
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"600{snapshot_idx}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(stock_idx))
                incumbent_predictions.append(float(9 - stock_idx))
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"{stock_idx:04d}.HK",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(9 - stock_idx))
                incumbent_predictions.append(float(stock_idx))

        route = _recent_positive_year_market_route(
            pd.DataFrame(rows),
            np.asarray(candidate_predictions, dtype="float64"),
            np.asarray(incumbent_predictions, dtype="float64"),
            gate_start="2024-07-07",
            primary_horizon="6m",
        )

        assert route == ("2025-01-01", {"ashare": 1.0})

    def test_recent_positive_window_market_route_selects_best_recent_start(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _recent_positive_window_market_route,
        )

        rows = []
        candidate_predictions = []
        incumbent_predictions = []
        for snapshot_date in pd.date_range("2025-01-01", periods=6, freq="MS"):
            candidate_good = snapshot_date >= pd.Timestamp("2025-03-01")
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"600{snapshot_date.month:02d}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(stock_idx if candidate_good else 9 - stock_idx))
                incumbent_predictions.append(float(9 - stock_idx if candidate_good else stock_idx))
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"{stock_idx:04d}.HK",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(9 - stock_idx))
                incumbent_predictions.append(float(stock_idx))

        route = _recent_positive_window_market_route(
            pd.DataFrame(rows),
            np.asarray(candidate_predictions, dtype="float64"),
            np.asarray(incumbent_predictions, dtype="float64"),
            gate_start="2024-12-15",
            primary_horizon="6m",
            min_snapshots=2,
        )

        assert route == ("2025-03-01", {"ashare": 1.0})

    def test_recent_positive_window_market_route_can_select_partial_weight(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _recent_positive_window_market_route,
        )

        rows = []
        candidate_predictions = []
        incumbent_predictions = []
        returns = [float(value) for value in range(10)]
        incumbent = [0.0, 1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 8.0, 9.0, 7.0]
        candidate = [0.0, 2.0, 1.0, 4.0, 3.0, 5.0, 7.0, 6.0, 9.0, 8.0]
        for snapshot_date in pd.date_range("2025-01-01", periods=4, freq="MS"):
            for stock_idx, forward_return in enumerate(returns):
                rows.append({
                    "ticker": f"600{snapshot_date.month:02d}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "forward_return_6m": forward_return,
                })
                candidate_predictions.append(candidate[stock_idx])
                incumbent_predictions.append(incumbent[stock_idx])

        route = _recent_positive_window_market_route(
            pd.DataFrame(rows),
            np.asarray(candidate_predictions, dtype="float64"),
            np.asarray(incumbent_predictions, dtype="float64"),
            gate_start="2024-12-15",
            primary_horizon="6m",
            min_snapshots=2,
        )

        assert route == ("2025-01-01", {"ashare": 0.5})

    def test_recent_positive_window_segment_route_selects_cap_bucket(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _recent_positive_window_segment_route,
        )

        rows = []
        candidate_predictions = []
        incumbent_predictions = []
        for snapshot_date in pd.date_range("2025-01-01", periods=4, freq="MS"):
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"600{snapshot_date.month:02d}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "market_cap_rmb": 5_000_000_000.0,
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(stock_idx))
                incumbent_predictions.append(float(9 - stock_idx))
            for stock_idx in range(10):
                rows.append({
                    "ticker": f"601{snapshot_date.month:02d}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "market_cap_rmb": 80_000_000_000.0,
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(float(9 - stock_idx))
                incumbent_predictions.append(float(stock_idx))

        route = _recent_positive_window_segment_route(
            pd.DataFrame(rows),
            np.asarray(candidate_predictions, dtype="float64"),
            np.asarray(incumbent_predictions, dtype="float64"),
            gate_start="2024-12-15",
            primary_horizon="6m",
            min_snapshots=2,
        )

        assert route is not None
        recent_start, routes = route
        assert recent_start == "2025-01-01"
        assert routes == [
            {
                "start_date": "2025-01-01",
                "market": "ashare",
                "cap_bucket": "small",
                "candidate_weight": 1.0,
            }
        ]

    def test_segment_routes_can_require_top20_excess_non_degradation(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _positive_segment_route,
            _recent_positive_window_segment_route,
        )

        rows = []
        candidate_predictions = []
        incumbent_predictions = []
        incumbent_pattern = [float(9 - idx) for idx in range(10)] + [
            float(39 - idx) for idx in range(10, 30)
        ]
        candidate_pattern = [float(idx) for idx in range(30)]
        candidate_pattern[0] = 100.0
        candidate_pattern[29] = 5.0

        for snapshot_date in pd.date_range("2025-01-01", periods=4, freq="MS"):
            for stock_idx in range(30):
                rows.append({
                    "ticker": f"600{snapshot_date.month:02d}{stock_idx:02d}",
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "market_cap_rmb": 5_000_000_000.0,
                    "forward_return_6m": float(stock_idx),
                })
                candidate_predictions.append(candidate_pattern[stock_idx])
                incumbent_predictions.append(incumbent_pattern[stock_idx])

        frame = pd.DataFrame(rows)
        candidate_array = np.asarray(candidate_predictions, dtype="float64")
        incumbent_array = np.asarray(incumbent_predictions, dtype="float64")

        assert _positive_segment_route(
            frame,
            candidate_array,
            incumbent_array,
            primary_horizon="6m",
            min_snapshots=2,
        ) is not None
        assert _positive_segment_route(
            frame,
            candidate_array,
            incumbent_array,
            primary_horizon="6m",
            min_snapshots=2,
            require_primary_top20_excess_non_degradation=True,
        ) is None
        assert _recent_positive_window_segment_route(
            frame,
            candidate_array,
            incumbent_array,
            gate_start="2024-12-15",
            primary_horizon="6m",
            min_snapshots=2,
        ) is not None
        assert _recent_positive_window_segment_route(
            frame,
            candidate_array,
            incumbent_array,
            gate_start="2024-12-15",
            primary_horizon="6m",
            min_snapshots=2,
            require_primary_top20_excess_non_degradation=True,
        ) is None

    def test_price_valuation_skeleton_estimates_market_cap_from_current_shares(
        self,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import data_prep

        monkeypatch.setattr(
            data_prep,
            "_current_market_cap_share_estimates",
            lambda: {"AAA": 1_000_000.0},
        )
        prices = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "date": ["2025-01-02", "2025-01-02"],
            "close": [10.0, 20.0],
        })

        valuations = data_prep._price_valuation_skeleton(
            prices,
            start=pd.Timestamp("2025-01-01").date(),
            end=pd.Timestamp("2025-01-31").date(),
        )

        aaa = valuations.loc[valuations["ticker"] == "AAA"].iloc[0]
        bbb = valuations.loc[valuations["ticker"] == "BBB"].iloc[0]
        assert aaa["market_cap_rmb"] == pytest.approx(10_000_000.0)
        assert pd.isna(bbb["market_cap_rmb"])

    def test_ml_ranker_scores_results_from_artifact(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef = [0.0] * len(feature_names)
        coef[feature_names.index("ticker_rankmean_6m")] = 1.0
        model_path = tmp_path / "model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": coef,
                "intercept": 0.0,
                "ticker_priors": {
                    "AAA": {"ticker_rankmean_6m": 0.8},
                    "BBB": {"ticker_rankmean_6m": -0.8},
                },
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(ticker="BBB", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[1].composite_score > results[0].composite_score

    def test_ml_ranker_scores_market_payload_member_artifact(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()

        def child_payload(intercept: float) -> dict[str, object]:
            return {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": [0.0] * len(feature_names),
                "intercept": intercept,
                "ticker_priors": {},
                "metadata": {},
            }

        model_path = tmp_path / "market_payload_model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": [],
                "ticker_priors": {},
                "metadata": {},
                "market_payload_members": [
                    {
                        "market_weights": {"ashare": 0.0, "hk": 1.0},
                        "payload": child_payload(10.0),
                    },
                    {
                        "market_weights": {"ashare": 1.0, "hk": 0.0},
                        "payload": child_payload(20.0),
                    },
                ],
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="0700.HK", name="T", market=Market.HK_SHARE),
                financials=Financials(ticker="0700.HK", period="snapshot"),
                valuation=ValuationMetrics(ticker="0700.HK", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[1].composite_score > results[0].composite_score

    def test_ml_ranker_scores_conditional_segment_artifact(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        candidate_coef = [0.0] * len(feature_names)
        candidate_coef[feature_names.index("value_score")] = 1.0

        def child_payload(coef: list[float]) -> dict[str, object]:
            return {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": coef,
                "intercept": 0.0,
                "ticker_priors": {},
                "metadata": {},
            }

        model_path = tmp_path / "conditional_model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": [],
                "ticker_priors": {},
                "metadata": {},
                "conditional_blend": {
                    "base_payload": child_payload([0.0] * len(feature_names)),
                    "candidate_payload": child_payload(candidate_coef),
                    "routes": [
                        {
                            "market": "ashare",
                            "cap_bucket": "small",
                            "board_bucket": "other",
                            "listing_bucket": "unknown",
                            "candidate_weight": 1.0,
                        }
                    ],
                },
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="SMALL", name="S", market=Market.A_SHARE),
                financials=Financials(ticker="SMALL", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="SMALL",
                    date="2026-01-01",
                    price=1.0,
                    market_cap_rmb=5_000_000_000.0,
                ),
                value_score=25.0,
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="LARGE", name="L", market=Market.A_SHARE),
                financials=Financials(ticker="LARGE", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="LARGE",
                    date="2026-01-01",
                    price=1.0,
                    market_cap_rmb=80_000_000_000.0,
                ),
                value_score=25.0,
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(25.0)
        assert results[1]._ml_ranker_raw_score == pytest.approx(0.0)
        assert results[0].composite_score > results[1].composite_score

    def test_ml_ranker_scores_results_from_ensemble_artifact(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef_a = [0.0] * len(feature_names)
        coef_b = [0.0] * len(feature_names)
        coef_a[feature_names.index("ticker_rankmean_6m")] = 1.0
        coef_b[feature_names.index("ticker_rankmean_6m")] = 0.5
        model_path = tmp_path / "model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "models": [
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_a,
                        "intercept": 0.0,
                    },
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_b,
                        "intercept": 0.0,
                    },
                ],
                "ticker_priors": {
                    "AAA": {"ticker_rankmean_6m": 0.8},
                    "BBB": {"ticker_rankmean_6m": -0.8},
                },
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(ticker="BBB", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(-0.6)
        assert results[1]._ml_ranker_raw_score == pytest.approx(0.6)
        assert results[1].composite_score > results[0].composite_score

    def test_ml_ranker_scores_temporal_artifact_by_valuation_date(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        early_coef = [0.0] * len(feature_names)
        recent_coef = [0.0] * len(feature_names)
        early_coef[feature_names.index("close")] = 1.0
        recent_coef[feature_names.index("close")] = 10.0

        def linear_payload(coef: list[float]) -> dict[str, object]:
            return {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "clip_low": [-100.0] * len(feature_names),
                "clip_high": [100.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
                "coef": coef,
                "intercept": 0.0,
                "ticker_priors": {},
                "metadata": {},
            }

        model_path = tmp_path / "model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": [],
                "temporal_payload_members": [
                    {"end_date": "2023-07-07", "payload": linear_payload(early_coef)},
                    {"start_date": "2023-07-07", "payload": linear_payload(recent_coef)},
                ],
                "ticker_priors": {},
                "metadata": {"target_horizon": "6m"},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="AAA",
                    date="2023-06-30",
                    price=2.0,
                ),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="BBB",
                    date="2023-07-07",
                    price=3.0,
                ),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(2.0)
        assert results[1]._ml_ranker_raw_score == pytest.approx(30.0)

    def test_ml_ranker_routes_market_specific_ensemble_models(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef_a = [0.0] * len(feature_names)
        coef_hk = [0.0] * len(feature_names)
        coef_a[feature_names.index("ticker_rankmean_6m")] = 1.0
        coef_hk[feature_names.index("ticker_rankmean_6m")] = -1.0
        model_path = tmp_path / "model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "models": [
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_a,
                        "intercept": 0.0,
                        "market": "ashare",
                    },
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_hk,
                        "intercept": 0.0,
                        "market": "hk",
                    },
                ],
                "ticker_priors": {
                    "AAA": {"ticker_rankmean_6m": 0.8},
                    "0700.HK": {"ticker_rankmean_6m": 0.8},
                },
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="0700.HK", name="T", market=Market.HK_SHARE),
                financials=Financials(ticker="0700.HK", period="snapshot"),
                valuation=ValuationMetrics(ticker="0700.HK", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(0.8)
        assert results[1]._ml_ranker_raw_score == pytest.approx(-0.8)

    def test_ml_ranker_routes_cap_segment_ensemble_models(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        model_path = tmp_path / "segment_model.json"
        base_model = {
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": [0.0] * len(feature_names),
            "market": "ashare",
        }
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "models": [
                    {**base_model, "intercept": 1.0, "cap_bucket": "small"},
                    {**base_model, "intercept": 10.0, "cap_bucket": "large"},
                ],
                "ticker_priors": {},
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="SMALL", name="S", market=Market.A_SHARE),
                financials=Financials(ticker="SMALL", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="SMALL",
                    date="snapshot",
                    price=1.0,
                    market_cap_rmb=5_000_000_000.0,
                ),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="LARGE", name="L", market=Market.A_SHARE),
                financials=Financials(ticker="LARGE", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="LARGE",
                    date="snapshot",
                    price=1.0,
                    market_cap_rmb=80_000_000_000.0,
                ),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(1.0)
        assert results[1]._ml_ranker_raw_score == pytest.approx(10.0)

    def test_ml_ranker_routes_board_segment_ensemble_models(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        base_model = {
            "clip_low": [-100.0] * len(feature_names),
            "clip_high": [100.0] * len(feature_names),
            "mean": [0.0] * len(feature_names),
            "scale": [1.0] * len(feature_names),
            "coef": [0.0] * len(feature_names),
            "market": "ashare",
            "cap_bucket": "large",
        }
        model_path = tmp_path / "board_segment_model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "models": [
                    {
                        **base_model,
                        "intercept": 1.0,
                        "board_bucket": "sh_main",
                        "listing_bucket": "600",
                    },
                    {
                        **base_model,
                        "intercept": 10.0,
                        "board_bucket": "star",
                        "listing_bucket": "688",
                    },
                ],
                "ticker_priors": {},
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="600000", name="S", market=Market.A_SHARE),
                financials=Financials(ticker="600000", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="600000",
                    date="snapshot",
                    price=1.0,
                    market_cap_rmb=80_000_000_000.0,
                ),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="688000", name="L", market=Market.A_SHARE),
                financials=Financials(ticker="688000", period="snapshot"),
                valuation=ValuationMetrics(
                    ticker="688000",
                    date="snapshot",
                    price=1.0,
                    market_cap_rmb=80_000_000_000.0,
                ),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(1.0)
        assert results[1]._ml_ranker_raw_score == pytest.approx(10.0)

    def test_ml_ranker_respects_weighted_ensemble_artifact(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        coef_a = [0.0] * len(feature_names)
        coef_b = [0.0] * len(feature_names)
        coef_a[feature_names.index("ticker_rankmean_6m")] = 1.0
        coef_b[feature_names.index("ticker_rankmean_6m")] = -1.0
        model_path = tmp_path / "model.json"
        model_path.write_text(
            json.dumps({
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_names": feature_names,
                "models": [
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_a,
                        "intercept": 0.0,
                        "weight": 0.75,
                    },
                    {
                        "clip_low": [-100.0] * len(feature_names),
                        "clip_high": [100.0] * len(feature_names),
                        "mean": [0.0] * len(feature_names),
                        "scale": [1.0] * len(feature_names),
                        "coef": coef_b,
                        "intercept": 0.0,
                        "weight": 0.25,
                    },
                ],
                "ticker_priors": {
                    "AAA": {"ticker_rankmean_6m": 1.0},
                    "BBB": {"ticker_rankmean_6m": -1.0},
                },
                "metadata": {},
            }),
            encoding="utf-8",
        )
        results = [
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(ticker="BBB", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(-0.5)
        assert results[1]._ml_ranker_raw_score == pytest.approx(0.5)

        payload = json.loads(model_path.read_text(encoding="utf-8"))
        for model in payload["models"]:
            model["weight"] = 0.0
        model_path.write_text(json.dumps(payload), encoding="utf-8")

        clear_model_cache()
        applied = score_results_with_ml_ranker(results, model_path=model_path)
        clear_model_cache()

        assert applied is True
        assert results[0]._ml_ranker_raw_score == pytest.approx(0.0)
        assert results[1]._ml_ranker_raw_score == pytest.approx(0.0)

    def test_ml_ranker_reload_reflects_latest_file_update(self, tmp_path: Path) -> None:
        from valueinvestor.data.models import (
            Company,
            Financials,
            Market,
            ScreeningResult,
            ValuationMetrics,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            clear_model_cache,
            expected_feature_names,
            score_results_with_ml_ranker,
        )

        feature_names = expected_feature_names()
        model_path = tmp_path / "model.json"

        def write_model(weight: float) -> None:
            coef = [0.0] * len(feature_names)
            coef[feature_names.index("ticker_rankmean_6m")] = weight
            model_path.write_text(
                json.dumps({
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "feature_names": feature_names,
                    "clip_low": [-100.0] * len(feature_names),
                    "clip_high": [100.0] * len(feature_names),
                    "mean": [0.0] * len(feature_names),
                    "scale": [1.0] * len(feature_names),
                    "coef": coef,
                    "intercept": 0.0,
                    "ticker_priors": {
                        "AAA": {"ticker_rankmean_6m": 0.8},
                        "BBB": {"ticker_rankmean_6m": -0.8},
                    },
                    "metadata": {},
                }),
                encoding="utf-8",
            )

        write_model(1.0)
        results = [
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(ticker="BBB", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        clear_model_cache()
        assert score_results_with_ml_ranker(results, model_path=model_path) is True
        first_scores = [result._ml_ranker_raw_score for result in results]

        time.sleep(0.01)
        write_model(-1.0)
        results = [
            ScreeningResult(
                company=Company(ticker="BBB", name="B", market=Market.A_SHARE),
                financials=Financials(ticker="BBB", period="snapshot"),
                valuation=ValuationMetrics(ticker="BBB", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
            ScreeningResult(
                company=Company(ticker="AAA", name="A", market=Market.A_SHARE),
                financials=Financials(ticker="AAA", period="snapshot"),
                valuation=ValuationMetrics(ticker="AAA", date="snapshot", price=1.0),
                composite_score=50.0,
            ),
        ]

        assert score_results_with_ml_ranker(results, model_path=model_path) is True
        second_scores = [result._ml_ranker_raw_score for result in results]

        assert first_scores != second_scores
        assert second_scores[1] < second_scores[0]

    def test_promotion_gate_split_keeps_holdout_untouched(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.scorer_improver.promotion_gate import (
            HoldoutGateConfig,
            make_holdout_split,
        )

        df = pd.DataFrame(
            {
                "snapshot_date": pd.date_range("2020-01-01", periods=36, freq="MS"),
                "ticker": ["T"] * 36,
            }
        )
        config = HoldoutGateConfig(
            holdout_months=6,
            embargo_days=45,
            min_gate_snapshots=4,
            min_train_snapshots=4,
        )

        source_path = tmp_path / "snapshots.parquet"
        source_path.write_bytes(b"snapshot-data")
        split = make_holdout_split(df, config=config, source_path=source_path)
        train_dates = pd.to_datetime(split.train["snapshot_date"])
        gate_dates = pd.to_datetime(split.gate["snapshot_date"])
        gate_start = pd.Timestamp(split.manifest["gate_start"])
        train_end = pd.Timestamp(split.manifest["train_end_exclusive"])

        assert train_dates.max() < train_end
        assert gate_dates.min() >= gate_start
        assert set(split.train.index).isdisjoint(split.gate.index)
        assert split.manifest["embargo_days"] == 45
        assert split.manifest["gate_snapshots"] >= 4
        assert split.manifest["source_path"] == str(source_path)
        assert split.manifest["source_fingerprint"].startswith("snapshots.parquet:13:")

    def test_promotion_gate_requires_holdout_6m_improvement(self) -> None:
        from valueinvestor.scorer_improver.promotion_gate import (
            HoldoutGateConfig,
            evaluate_promotion_gate,
        )

        incumbent = {
            "1m": {"spearman_rho": 0.10, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "3m": {"spearman_rho": 0.20, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "6m": {"spearman_rho": 0.30, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
        }
        candidate = {
            "1m": {"spearman_rho": 0.11, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "3m": {"spearman_rho": 0.21, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "6m": {"spearman_rho": 0.3004, "hit_rate_top20": 0.51, "mean_excess_return": 0.02},
        }

        result = evaluate_promotion_gate(
            candidate_metrics=candidate,
            incumbent_metrics=incumbent,
            manifest={"gate_snapshots": 8},
            config=HoldoutGateConfig(min_6m_delta=0.001),
        )

        assert result.accepted is False
        assert result.reason == "6m delta below gate"

    def test_promotion_gate_requires_absolute_primary_rho_floor(self) -> None:
        from valueinvestor.scorer_improver.promotion_gate import (
            HoldoutGateConfig,
            evaluate_promotion_gate,
        )

        incumbent = {
            "1m": {"spearman_rho": 0.05, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "3m": {"spearman_rho": 0.06, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "6m": {"spearman_rho": 0.07, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
        }
        candidate = {
            "1m": {"spearman_rho": 0.12, "hit_rate_top20": 0.55, "mean_excess_return": 0.02},
            "3m": {"spearman_rho": 0.20, "hit_rate_top20": 0.55, "mean_excess_return": 0.02},
            "6m": {"spearman_rho": 0.29, "hit_rate_top20": 0.55, "mean_excess_return": 0.02},
        }

        result = evaluate_promotion_gate(
            candidate_metrics=candidate,
            incumbent_metrics=incumbent,
            manifest={"gate_snapshots": 8},
            config=HoldoutGateConfig(min_6m_delta=0.001, min_primary_rho=0.307),
        )

        assert result.accepted is False
        assert result.reason == "6m rho below absolute gate"

    def test_promotion_gate_rejects_regime_concentrated_degradation(self) -> None:
        from valueinvestor.scorer_improver.promotion_gate import (
            HoldoutGateConfig,
            evaluate_promotion_gate,
        )

        metrics = {
            "1m": {"spearman_rho": 0.20, "hit_rate_top20": 0.60, "mean_excess_return": 0.02},
            "3m": {"spearman_rho": 0.20, "hit_rate_top20": 0.60, "mean_excess_return": 0.02},
            "6m": {"spearman_rho": 0.31, "hit_rate_top20": 0.60, "mean_excess_return": 0.02},
        }
        incumbent = {
            "1m": {"spearman_rho": 0.10, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "3m": {"spearman_rho": 0.10, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
            "6m": {"spearman_rho": 0.30, "hit_rate_top20": 0.50, "mean_excess_return": 0.01},
        }

        result = evaluate_promotion_gate(
            candidate_metrics=metrics,
            incumbent_metrics=incumbent,
            manifest={"gate_snapshots": 8},
            config=HoldoutGateConfig(min_6m_delta=0.001),
            regime_diagnostics={
                "market": [
                    {"label": "ashare", "deltas": {"6m": 0.02}},
                    {"label": "hk", "deltas": {"6m": -0.08}},
                ]
            },
        )

        assert result.accepted is False
        assert result.reason == "regime market 6m degradation"

    def test_asof_feature_frame_uses_past_rows_only(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import asof_feature_frame

        features = pd.DataFrame({
            "ticker": ["AAA", "AAA"],
            "date": ["2020-01-01", "2020-03-01"],
            "pe_ratio": [10.0, 20.0],
        })
        tickers = pd.Series(["AAA", "AAA", "BBB"])
        snapshots = pd.Series([
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-04-01"),
            pd.Timestamp("2020-04-01"),
        ])

        result = asof_feature_frame(features, ("pe_ratio",), tickers, snapshots)

        assert result.loc[0, "pe_ratio"] == pytest.approx(10.0)
        assert result.loc[1, "pe_ratio"] == pytest.approx(20.0)
        assert pd.isna(result.loc[2, "pe_ratio"])

        lagged = asof_feature_frame(features, ("pe_ratio",), tickers.iloc[[1]], snapshots.iloc[[1]], lag_days=45)
        assert lagged.loc[0, "pe_ratio"] == pytest.approx(10.0)

    def test_ml_snapshot_cache_requires_current_manifest(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.ml_trainer import prepare_ml_training_snapshots

        stale_path = tmp_path / "snapshots.parquet"
        pd.DataFrame({
            "ticker": ["AAA"],
            "snapshot_date": [pd.Timestamp("2020-01-01")],
            "target_rank_1w": [0.0],
            "target_rank_1w_market": [0.0],
            "target_rank_1m": [0.0],
            "target_rank_3m": [0.0],
            "target_rank_6m": [0.0],
        }).to_parquet(stale_path, index=False)

        with pytest.raises(RuntimeError, match="has no manifest"):
            prepare_ml_training_snapshots(
                output_path=stale_path,
                start_date=pd.Timestamp("2020-01-01").date(),
                end_date=pd.Timestamp("2020-12-31").date(),
            )

    def test_ml_snapshot_manifest_allows_current_cache(self, tmp_path: Path) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _write_snapshot_manifest,
            prepare_ml_training_snapshots,
        )

        snapshot_path = tmp_path / "snapshots.parquet"
        start = pd.Timestamp("2020-01-01").date()
        end = pd.Timestamp("2020-12-31").date()
        df = pd.DataFrame({
            "ticker": ["AAA"],
            "snapshot_date": [pd.Timestamp("2020-01-01")],
            "target_rank_1w": [0.0],
            "target_rank_1w_market": [0.0],
            "target_rank_1m": [0.0],
            "target_rank_3m": [0.0],
            "target_rank_6m": [0.0],
        })
        df.to_parquet(snapshot_path, index=False)
        _write_snapshot_manifest(
            snapshot_path,
            df,
            snapshot_frequency="daily",
            ground_truth_path=tmp_path / "ground_truth.parquet",
            start_date=start,
            end_date=end,
        )

        cached = prepare_ml_training_snapshots(
            output_path=snapshot_path,
            ground_truth_path=tmp_path / "ground_truth.parquet",
            start_date=start,
            end_date=end,
        )

        assert len(cached) == 1

    def test_ml_snapshot_manifest_ignores_trainer_search_fingerprint(
        self,
        tmp_path: Path,
    ) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _snapshot_manifest_path,
            _write_snapshot_manifest,
            prepare_ml_training_snapshots,
        )

        snapshot_path = tmp_path / "snapshots.parquet"
        start = pd.Timestamp("2020-01-01").date()
        end = pd.Timestamp("2020-12-31").date()
        df = pd.DataFrame({
            "ticker": ["AAA"],
            "snapshot_date": [pd.Timestamp("2020-01-01")],
            "target_rank_1w": [0.0],
            "target_rank_1w_market": [0.0],
            "target_rank_1m": [0.0],
            "target_rank_3m": [0.0],
            "target_rank_6m": [0.0],
        })
        df.to_parquet(snapshot_path, index=False)
        _write_snapshot_manifest(
            snapshot_path,
            df,
            snapshot_frequency="daily",
            ground_truth_path=tmp_path / "ground_truth.parquet",
            start_date=start,
            end_date=end,
        )
        manifest_path = _snapshot_manifest_path(snapshot_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_fingerprints"]["ml_trainer"] = "stale-search-code"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        cached = prepare_ml_training_snapshots(
            output_path=snapshot_path,
            ground_truth_path=tmp_path / "ground_truth.parquet",
            start_date=start,
            end_date=end,
        )

        assert len(cached) == 1

    def test_point_in_time_source_validation_rejects_future_only_features(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _validate_point_in_time_feature_sources,
        )

        valuations = pd.DataFrame({
            "ticker": ["AAA"],
            "date": [pd.Timestamp("2026-01-01")],
            "pe_ratio": [10.0],
        })

        with pytest.raises(RuntimeError, match="Point-in-time feature sources"):
            _validate_point_in_time_feature_sources(
                valuations=valuations,
                financials=pd.DataFrame(),
                end_date=pd.Timestamp("2025-01-01").date(),
            )

    def test_rolling_ticker_priors_use_only_known_past_rows(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _attach_asof_ticker_priors

        df = pd.DataFrame({
            "ticker": ["AAA", "AAA", "AAA"],
            "snapshot_date": pd.to_datetime(["2020-01-01", "2020-03-01", "2020-07-01"]),
            "forward_return_1w": [0.01, 0.03, 0.09],
            "forward_return_1m": [0.10, 0.30, 0.90],
            "forward_return_3m": [0.20, 0.40, 0.80],
            "forward_return_6m": [0.50, 0.70, 0.60],
            "target_rank_1w": [0.1, 0.3, 0.9],
            "target_rank_1m": [0.1, 0.3, 0.9],
            "target_rank_3m": [0.2, 0.4, 0.8],
            "target_rank_6m": [0.5, 0.7, 0.6],
        })

        result = _attach_asof_ticker_priors(df, df, embargo_days=0)

        assert result.loc[0, "_prior_ticker_mean_1m"] == pytest.approx(0.0)
        assert result.loc[1, "_prior_ticker_mean_1m"] == pytest.approx(0.10)
        assert result.loc[2, "_prior_ticker_mean_1m"] == pytest.approx(0.20)
        assert result.loc[2, "_prior_ticker_mean_6m"] == pytest.approx(0.50)

    def test_ml_trainer_walk_forward_decision_rejects_bad_mean_6m(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _walk_forward_decision

        result = _walk_forward_decision(
            [
                {"1m": 0.01, "3m": 0.00, "6m": -0.02},
                {"1m": 0.00, "3m": 0.01, "6m": 0.00},
            ],
            min_6m_delta=0.001,
            max_horizon_degradation=0.01,
        )

        assert result["accepted"] is False
        assert result["reason"] == "mean 6m delta below walk-forward gate"

    def test_ml_trainer_walk_forward_decision_allows_one_bounded_bad_fold(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _walk_forward_decision

        fold_deltas = [
            {"1w": 0.09, "1m": 0.03, "3m": -0.02, "6m": -0.08},
            {"1w": -0.01, "1m": -0.04, "3m": -0.03, "6m": 0.07},
            {"1w": 0.05, "1m": 0.11, "3m": 0.18, "6m": 0.18},
            {"1w": 0.04, "1m": 0.05, "3m": 0.02, "6m": 0.04},
        ]

        bounded_result = _walk_forward_decision(
            fold_deltas,
            min_6m_delta=0.001,
            max_horizon_degradation=0.10,
        )
        strict_result = _walk_forward_decision(
            fold_deltas,
            min_6m_delta=0.001,
            max_horizon_degradation=0.01,
        )

        assert bounded_result["accepted"] is True
        assert bounded_result["min_deltas"]["6m"] == pytest.approx(-0.08)
        assert strict_result["accepted"] is False
        assert strict_result["reason"] == "fold 6m degradation"

    def test_walk_forward_blend_search_fits_each_fold_once_and_selects_best(
        self,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import ml_trainer

        rows = []
        for snapshot_date, reverse in (
            ("2020-01-02", False),
            ("2021-01-04", True),
        ):
            for index in range(10):
                rank = float(index)
                rows.append(
                    {
                        "ticker": f"T{index:02d}",
                        "snapshot_date": snapshot_date,
                        "composite_score": float(9 - index) if reverse else rank,
                        "forward_return_1w": rank,
                        "forward_return_1m": rank,
                        "forward_return_3m": rank,
                        "forward_return_6m": rank,
                        "target_rank_1w": rank,
                        "target_rank_1m": rank,
                        "target_rank_3m": rank,
                        "target_rank_6m": rank,
                    }
                )
        snapshots = pd.DataFrame(rows)
        fit_calls = []

        def fake_fit(*args, **kwargs):
            fit_calls.append(1)
            return {}, {}, "numpy", False

        monkeypatch.setattr(ml_trainer, "_fit_candidate_model", fake_fit)
        monkeypatch.setattr(
            ml_trainer,
            "_predict_from_linear_model",
            lambda frame, *args, **kwargs: np.arange(len(frame), dtype="float64"),
        )

        result = ml_trainer._walk_forward_validate_candidate(
            snapshots,
            [
                {
                    "index": 1,
                    "train_end_exclusive": "2020-07-01",
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-01-31",
                }
            ],
            target_name="target_rank_6m",
            ridge_lambda=1.0,
            backend="numpy",
            max_rows=None,
            prior_strategy="no_ticker_priors",
            model_kind="ridge",
            feature_names=[],
            incumbent_payload=None,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            blend_weights=(0.5, 1.0),
        )

        assert len(fit_calls) == 1
        assert result["accepted"] is True
        assert result["blend_weight"] == pytest.approx(1.0)
        assert result["blend_scale"] == "snapshot_rank"
        assert result["blend_candidates_evaluated"] == 2

    def test_walk_forward_selects_practical_factor_candidate(
        self,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import ml_trainer

        snapshots = pd.DataFrame(
            {
                "ticker": [f"T{index:02d}" for index in range(20)],
                "snapshot_date": ["2020-01-02"] * 10 + ["2021-01-04"] * 10,
                "composite_score": list(reversed(range(10))) * 2,
                "forward_return_1w": list(range(10)) * 2,
                "forward_return_1m": list(range(10)) * 2,
                "forward_return_3m": list(range(10)) * 2,
                "forward_return_6m": list(range(10)) * 2,
                "target_rank_1w": list(range(10)) * 2,
                "target_rank_1m": list(range(10)) * 2,
                "target_rank_3m": list(range(10)) * 2,
                "target_rank_6m": list(range(10)) * 2,
            }
        )
        monkeypatch.setattr(
            ml_trainer,
            "_fit_candidate_model",
            lambda *args, **kwargs: ({}, {}, "numpy", False),
        )
        monkeypatch.setattr(
            ml_trainer,
            "_predict_from_linear_model",
            lambda frame, *args, **kwargs: np.arange(len(frame), dtype="float64"),
        )
        factor_calls = []

        def select_factor(*args, **kwargs):
            factor_calls.append(kwargs)
            rhos = {horizon: 2.0 for horizon in ml_trainer.EVAL_HORIZONS}
            return {
                "accepted": True,
                "blend_weight": 1.0,
                "mean_deltas": rhos,
                "median_deltas": rhos,
                "min_deltas": rhos,
                "application_factor_template": "test_factor",
                "application_factor_components": [{"signal": "roe_de_crowding", "weight": 1.0}],
                "application_factor_min_rhos": rhos,
                "application_factor_mean_rhos": rhos,
            }

        monkeypatch.setattr(ml_trainer, "_select_practical_factor_blend", select_factor)

        result = ml_trainer._walk_forward_validate_candidate(
            snapshots,
            [
                {
                    "index": 1,
                    "train_end_exclusive": "2020-07-01",
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-01-31",
                }
            ],
            target_name="target_rank_6m",
            ridge_lambda=1.0,
            backend="numpy",
            max_rows=None,
            prior_strategy="no_ticker_priors",
            model_kind="ridge",
            feature_names=[],
            incumbent_payload=None,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            blend_weights=(1.0,),
        )

        assert len(factor_calls) == 1
        assert factor_calls[0]["base_blend_weight"] == pytest.approx(1.0)
        assert result["application_factor_template"] == "test_factor"

        def select_weak_factor(*args, **kwargs):
            rhos = {horizon: 1.001 for horizon in ml_trainer.EVAL_HORIZONS}
            return {
                "accepted": True,
                "blend_weight": 1.0,
                "mean_deltas": rhos,
                "median_deltas": rhos,
                "min_deltas": rhos,
                "application_factor_template": "weak_factor",
                "application_factor_components": [{"signal": "roe_de_crowding", "weight": 1.0}],
                "application_factor_min_rhos": rhos,
                "application_factor_mean_rhos": rhos,
            }

        monkeypatch.setattr(ml_trainer, "_select_practical_factor_blend", select_weak_factor)
        fallback = ml_trainer._walk_forward_validate_candidate(
            snapshots,
            [
                {
                    "index": 1,
                    "train_end_exclusive": "2020-07-01",
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-01-31",
                }
            ],
            target_name="target_rank_6m",
            ridge_lambda=1.0,
            backend="numpy",
            max_rows=None,
            prior_strategy="no_ticker_priors",
            model_kind="ridge",
            feature_names=[],
            incumbent_payload=None,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            blend_weights=(1.0,),
        )

        assert "application_factor_template" not in fallback

    def test_walk_forward_selects_stronger_residual_candidate(
        self,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import ml_trainer

        snapshots = pd.DataFrame(
            {
                "ticker": [f"T{index:02d}" for index in range(20)],
                "snapshot_date": ["2020-01-02"] * 10 + ["2021-01-04"] * 10,
                "quality_score": list(range(10)) * 2,
                "composite_score": list(reversed(range(10))) * 2,
                "forward_return_1w": list(range(10)) * 2,
                "forward_return_1m": list(range(10)) * 2,
                "forward_return_3m": list(range(10)) * 2,
                "forward_return_6m": list(range(10)) * 2,
                "target_rank_1w": list(range(10)) * 2,
                "target_rank_1m": list(range(10)) * 2,
                "target_rank_3m": list(range(10)) * 2,
                "target_rank_6m": list(range(10)) * 2,
            }
        )

        monkeypatch.setattr(
            ml_trainer,
            "_fit_candidate_model",
            lambda *args, **kwargs: ({}, {}, "numpy", False),
        )
        monkeypatch.setattr(
            ml_trainer,
            "_predict_from_linear_model",
            lambda frame, *args, **kwargs: np.arange(len(frame), dtype="float64"),
        )

        def overlay_result(marker: str, mean_delta: float) -> dict[str, object]:
            deltas = {horizon: mean_delta for horizon in ml_trainer.EVAL_HORIZONS}
            return {
                "accepted": True,
                "blend_weight": 1.0,
                "mean_deltas": deltas,
                "median_deltas": deltas,
                "min_deltas": deltas,
                marker: marker,
            }

        monkeypatch.setattr(
            ml_trainer,
            "_select_application_residual_blend",
            lambda *args, **kwargs: overlay_result("application_residual_signal", 20.0),
        )

        result = ml_trainer._walk_forward_validate_candidate(
            snapshots,
            [
                {
                    "index": 1,
                    "train_end_exclusive": "2020-07-01",
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-01-31",
                }
            ],
            target_name="target_rank_6m",
            ridge_lambda=1.0,
            backend="numpy",
            max_rows=None,
            prior_strategy="no_ticker_priors",
            model_kind="ridge",
            feature_names=[],
            incumbent_payload=None,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            blend_weights=(1.0,),
            quality_residual_weights=(1.0, 0.5),
        )

        assert result["application_residual_signal"] == "application_residual_signal"

        rejected_residual = overlay_result("application_residual_signal", 20.0)
        rejected_residual["accepted"] = False
        monkeypatch.setattr(
            ml_trainer,
            "_select_application_residual_blend",
            lambda *args, **kwargs: rejected_residual,
        )
        fallback = ml_trainer._walk_forward_validate_candidate(
            snapshots,
            [
                {
                    "index": 1,
                    "train_end_exclusive": "2020-07-01",
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-01-31",
                }
            ],
            target_name="target_rank_6m",
            ridge_lambda=1.0,
            backend="numpy",
            max_rows=None,
            prior_strategy="no_ticker_priors",
            model_kind="ridge",
            feature_names=[],
            incumbent_payload=None,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            blend_weights=(1.0,),
            quality_residual_weights=(1.0, 0.5),
        )

        assert fallback["accepted"] is True
        assert "application_residual_signal" not in fallback

    def test_quality_residual_selection_maximizes_worst_fold_rho(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _select_quality_residual_blend,
        )

        fold_predictions = []
        for reverse_candidate in (False, True):
            rank = np.arange(10, dtype="float64")
            centered_rank = np.linspace(-0.5, 0.5, 10)
            validation = pd.DataFrame(
                {
                    "ticker": [f"T{index:02d}" for index in range(10)],
                    "snapshot_date": ["2025-01-02"] * 10,
                    "quality_score": rank[::-1],
                    "forward_return_1w": rank,
                    "forward_return_1m": rank,
                    "forward_return_3m": rank,
                    "forward_return_6m": rank,
                }
            )
            candidate = centered_rank[::-1] if reverse_candidate else centered_rank
            incumbent = centered_rank[::-1]
            fold_predictions.append((validation, candidate, incumbent))

        selected = _select_quality_residual_blend(
            fold_predictions,
            base_blend_weight=1.0,
            residual_weights=(1.0, 0.5),
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            primary_horizon="6m",
            blend_candidates_evaluated=1,
        )

        assert selected["accepted"] is True
        assert selected["quality_residual_candidate_weight"] == pytest.approx(0.5)
        assert selected["quality_residual_min_rhos"]["6m"] >= 0.0
        assert selected["quality_residual_candidates_evaluated"] == 2

    def test_application_residual_selection_chooses_practical_book_yield(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _select_application_residual_blend,
        )

        fold_predictions = []
        for snapshot_date in ("2024-01-02", "2024-07-02"):
            rank = np.arange(10, dtype="float64")
            validation = pd.DataFrame(
                {
                    "ticker": [f"T{index:02d}" for index in range(10)],
                    "snapshot_date": [snapshot_date] * 10,
                    "quality_score": rank,
                    "pb_ratio": rank[::-1] + 1.0,
                    "forward_return_1w": rank,
                    "forward_return_1m": rank,
                    "forward_return_3m": rank,
                    "forward_return_6m": rank,
                }
            )
            candidate = np.linspace(0.5, -0.5, 10)
            incumbent = candidate.copy()
            fold_predictions.append((validation, candidate, incumbent))

        selected = _select_application_residual_blend(
            fold_predictions,
            base_blend_weight=1.0,
            residual_weights=(1.0, 0.5),
            residual_signals=(
                ("quality_de_crowding", "quality_score", -1.0),
                ("book_yield", "pb_ratio", -1.0),
            ),
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            primary_horizon="6m",
            blend_candidates_evaluated=1,
            require_primary_top20_excess_non_degradation=True,
        )

        assert selected["accepted"] is True
        assert selected["application_residual_signal"] == "book_yield"
        assert selected["application_residual_candidate_weight"] == pytest.approx(0.5)
        assert selected["application_residual_fold_practical_passes"] == [True, True]

    def test_application_factor_selection_uses_purged_fold_templates(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_practical_factor_blend

        factor_templates = (
            ("equal", (0.10, 0.15, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.15)),
            ("balanced", (0.10, 0.18, 0.0, 0.0, 0.0, 0.135, 0.09, 0.135, 0.225, 0.135)),
            ("value_momentum", (0.10, 0.135, 0.0, 0.0, 0.0, 0.18, 0.045, 0.135, 0.27, 0.135)),
        )

        fold_predictions = []
        for snapshot_date in ("2024-01-02", "2024-07-02"):
            rank = np.arange(30, dtype="float64")
            reverse_rank = rank[::-1]
            validation = pd.DataFrame(
                {
                    "ticker": [f"T{index:02d}" for index in range(30)],
                    "snapshot_date": [snapshot_date] * 30,
                    "quality_score": reverse_rank,
                    "revenue": rank + 100.0,
                    "gross_margin": np.full(30, 0.25),
                    "total_assets": rank + 200.0,
                    "roe": reverse_rank / 100.0,
                    "roa": reverse_rank / 200.0,
                    "pb_ratio": reverse_rank + 1.0,
                    "ps_ratio": reverse_rank + 1.0,
                    "total_liabilities": rank + 1.0,
                    "market_cap_rmb": np.full(30, 100.0),
                    "relative_return_63d": rank,
                    "current_ratio": reverse_rank + 1.0,
                    "forward_return_1w": rank,
                    "forward_return_1m": rank,
                    "forward_return_3m": rank,
                    "forward_return_6m": rank,
                }
            )
            candidate = np.linspace(-0.5, 0.5, 30)
            incumbent = candidate[::-1]
            fold_predictions.append((validation, candidate, incumbent))

        selected = _select_practical_factor_blend(
            fold_predictions,
            base_blend_weight=1.0,
            factor_templates=factor_templates,
            min_6m_delta=0.001,
            max_horizon_degradation=2.0,
            primary_horizon="6m",
            blend_candidates_evaluated=1,
            require_primary_top20_excess_non_degradation=True,
        )

        assert selected is not None
        assert selected["accepted"] is True
        assert selected["application_factor_template"] == "value_momentum"
        assert selected["application_factor_candidates_evaluated"] == 3
        assert selected["application_factor_fold_practical_passes"] == [True, True]

    def test_ml_trainer_target_values_supports_one_week_target(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _target_values

        snapshots = pd.DataFrame({"target_rank_1w": [-0.5, 0.25, 0.75]})

        values = _target_values(snapshots, "target_rank_1w")

        assert values.tolist() == [-0.5, 0.25, 0.75]

    def test_ml_trainer_target_values_supports_legacy_six_month_search_targets(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _target_values

        snapshots = pd.DataFrame({
            "ticker": ["000001", "000002", "0005.HK"],
            "snapshot_date": [pd.Timestamp("2024-01-01").date()] * 3,
            "forward_return_1m": [0.1, 0.2, 0.3],
            "forward_return_3m": [0.2, 0.4, 0.6],
            "forward_return_6m": [0.3, 0.6, 0.9],
            "target_rank_1m": [-0.5, 0.0, 0.5],
            "target_rank_3m": [-0.25, 0.0, 0.25],
            "target_rank_6m": [-1.0, 0.0, 1.0],
        })

        assert _target_values(snapshots, "target_rank_6m_soft").tolist() == pytest.approx(
            [-1.0, 0.0, 1.0]
        )
        assert _target_values(snapshots, "target_rank_6m_extreme").tolist() == pytest.approx(
            [-1.0, 0.0, 1.0]
        )
        assert _target_values(snapshots, "target_rank_weighted_802").tolist() == pytest.approx(
            [-0.85, 0.0, 0.85]
        )
        assert _target_values(snapshots, "target_rank_weighted_901").tolist() == pytest.approx(
            [-0.925, 0.0, 0.925]
        )
        assert _target_values(snapshots, "target_rank_weighted_8515").tolist() == pytest.approx(
            [-0.8875, 0.0, 0.8875]
        )
        assert _target_values(snapshots, "target_rank_weighted_7525").tolist() == pytest.approx(
            [-0.8125, 0.0, 0.8125]
        )
        assert _target_values(snapshots, "target_rank_6m_market").tolist() == pytest.approx(
            [-1.0 / 3.0, 1.0 / 3.0, float("nan")],
            nan_ok=True,
        )
        assert _target_values(
            snapshots,
            "target_rank_weighted_703_market",
        ).tolist() == pytest.approx(
            [-1.0 / 3.0, 1.0 / 3.0, float("nan")],
            nan_ok=True,
        )

    def test_ml_trainer_payload_with_ticker_priors_replaces_gate_priors(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _payload_with_ticker_priors

        old_priors = {"OLD": {"ticker_rankmean_6m": 1.0}}
        payload = {
            "schema_version": "test",
            "feature_names": ["ticker_rankmean_6m"],
            "ticker_priors": old_priors,
            "temporal_payload_members": [
                {
                    "start_date": "2025-01-01",
                    "payload": {
                        "schema_version": "test",
                        "feature_names": ["ticker_rankmean_6m"],
                        "ticker_priors": old_priors,
                    },
                },
            ],
        }
        train_priors = {"NEW": {"ticker_rankmean_6m": -1.0}}

        patched = _payload_with_ticker_priors(payload, train_priors)

        assert patched is not None
        assert patched["ticker_priors"] == train_priors
        nested_payload = patched["temporal_payload_members"][0]["payload"]
        assert nested_payload["ticker_priors"] == train_priors
        original_nested_payload = payload["temporal_payload_members"][0]["payload"]
        assert original_nested_payload["ticker_priors"] == old_priors
        assert payload["ticker_priors"] == old_priors

    def test_single_temporal_member_payload_rejects_cached_metrics(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _payload_cached_metrics_are_safe,
            _payload_runtime_includes_evaluation_labels,
        )

        child_payload = {
            "schema_version": "test",
            "feature_names": ["close"],
            "ticker_priors": {},
        }
        bounded_single_member = {
            "schema_version": "test",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {"metrics": {"6m": {"spearman_rho": 0.3}}},
            "temporal_payload_members": [
                {"start_date": "2025-01-01", "payload": child_payload},
            ],
        }
        two_member_payload = {
            **bounded_single_member,
            "temporal_payload_members": [
                {"end_date": "2025-01-01", "payload": child_payload},
                {"start_date": "2025-01-01", "payload": child_payload},
            ],
        }

        assert not _payload_cached_metrics_are_safe(bounded_single_member)
        assert _payload_cached_metrics_are_safe(two_member_payload)

        ordinary_refit = {
            **child_payload,
            "metadata": {"full_fit_refit": True},
        }
        application_refit = {
            **child_payload,
            "metadata": {"application_refit": True},
        }
        deployment_refit = {
            **child_payload,
            "metadata": {"deployment_refit": {"frozen_after_outer_evaluation": True}},
        }
        full_refit_protocol = {
            **child_payload,
            "metadata": {"evaluation_protocol": "full_refit_current_24m_monthly_gate"},
        }
        nested_application_refit = {
            **child_payload,
            "rank_payload_members": [{"weight": 1.0, "payload": application_refit}],
        }

        assert _payload_cached_metrics_are_safe(ordinary_refit)
        assert not _payload_runtime_includes_evaluation_labels(ordinary_refit)
        for unsafe_payload in (
            application_refit,
            deployment_refit,
            full_refit_protocol,
            nested_application_refit,
        ):
            assert _payload_runtime_includes_evaluation_labels(unsafe_payload)
            assert not _payload_cached_metrics_are_safe(unsafe_payload)

    def test_ml_ranker_payload_uses_row_priors_when_marked_rolling(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["AAA"],
            "_prior_ticker_mean_6m": [0.8],
        })
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["ticker_mean_6m"],
            "ticker_priors": {"AAA": {"ticker_mean_6m": 0.1}},
            "metadata": {"uses_rolling_row_priors": True},
            "models": [{
                "clip_low": [-10.0],
                "clip_high": [10.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }

        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([0.8])

    def test_ml_ranker_payload_member_blend_keeps_member_schemas(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["AAA"],
            "close": [10.0],
            "price_return_5d": [0.2],
        })
        core_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        short_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["price_return_5d"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [-1.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [100.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {},
            "payload_members": [
                {"weight": 0.75, "payload": core_payload},
                {"weight": 0.25, "payload": short_payload},
            ],
        }

        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([12.5])

    def test_ml_ranker_market_payload_member_blend_routes_by_market(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["000001.SZ", "0700.HK"],
            "close": [10.0, 20.0],
            "price_return_5d": [0.2, 0.3],
        })
        anchor_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        candidate_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["price_return_5d"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [-1.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [100.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {},
            "market_payload_members": [
                {
                    "market_weights": {"ashare": 0.9, "hk": 0.5},
                    "payload": anchor_payload,
                },
                {
                    "market_weights": {"ashare": 0.1, "hk": 0.5},
                    "payload": candidate_payload,
                },
            ],
        }

        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([11.0, 25.0])

    def test_ml_ranker_cross_sectional_prediction_keeps_whole_snapshot_for_market_models(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB", "0700.HK", "AAA"],
            "snapshot_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "value_score": [10.0, 20.0, 30.0, 40.0],
        })
        model = {
            "clip_low": [-1.0],
            "clip_high": [1.0],
            "mean": [0.0],
            "scale": [1.0],
            "coef": [1.0],
            "intercept": 0.0,
            "weight": 1.0,
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["cs_rank_value_score"],
            "ticker_priors": {},
            "metadata": {},
            "models": [
                {**model, "market": "ashare"},
                {**model, "market": "hk"},
            ],
        }

        predictions = predict_ml_ranker_payload(frame, payload, chunk_size=2)

        assert predictions.tolist() == pytest.approx([-0.5, 0.0, 0.5, 0.0])

    def test_ml_ranker_temporal_payload_routes_by_snapshot_date(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "snapshot_date": ["2023-06-30", "2023-07-07"],
            "close": [2.0, 3.0],
        })
        early_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        recent_payload = {
            **early_payload,
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [10.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {"backend": "temporal_payload_blend"},
            "temporal_payload_members": [
                {"end_date": "2023-07-07", "payload": early_payload},
                {"start_date": "2023-07-07", "payload": recent_payload},
            ],
        }

        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([2.0, 30.0])

    def test_ml_ranker_single_member_temporal_payload_delegates_to_member(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import predict_ml_ranker_payload

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "snapshot_date": ["2024-01-01", "2026-01-01"],
            "close": [2.0, 3.0],
        })
        child_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [10.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {"backend": "temporal_payload_blend"},
            "temporal_payload_members": [
                {"start_date": "2025-01-01", "payload": child_payload},
            ],
        }

        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([20.0, 30.0])

    def test_runtime_payload_compaction_drops_recursive_training_diagnostics(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _compact_runtime_payload,
            predict_ml_ranker_payload,
        )

        frame = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "snapshot_date": ["2023-06-30", "2023-07-07"],
            "close": [2.0, 3.0],
        })
        child_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {
                "backend": "ridge",
                "metrics": {"6m": {"spearman_rho": 0.2}},
                "promotion_gate_attempts": [{"large": "diagnostic"}],
            },
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {
                "backend": "temporal_payload_blend",
                "metrics": {"6m": {"spearman_rho": 0.3}},
                "strict_outer_gate": True,
                "promotion_gate_manifest": {"gate_start": "2023-07-07"},
                "promotion_gate_attempts": [{"large": "diagnostic"}],
                "promotion_gate": {"regime_diagnostics": [{"large": "diagnostic"}]},
            },
            "temporal_payload_members": [
                {"end_date": "2023-07-07", "payload": child_payload},
                {"start_date": "2023-07-07", "payload": child_payload},
            ],
        }

        compacted = _compact_runtime_payload(payload)
        assert isinstance(compacted, dict)
        assert compacted["metadata"] == {
            "backend": "temporal_payload_blend",
            "metrics": {"6m": {"spearman_rho": 0.3}},
            "strict_outer_gate": True,
            "promotion_gate_manifest": {"gate_start": "2023-07-07"},
        }
        nested_metadata = compacted["temporal_payload_members"][0]["payload"]["metadata"]
        assert nested_metadata == {
            "backend": "ridge",
            "metrics": {"6m": {"spearman_rho": 0.2}},
        }
        assert predict_ml_ranker_payload(frame, compacted).tolist() == pytest.approx(
            predict_ml_ranker_payload(frame, payload).tolist()
        )

    def test_runtime_payload_compaction_collapses_recent_segment_temporal_payload(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _compact_runtime_payload,
            _conditional_blend_payload,
            predict_ml_ranker_payload,
        )

        frame = pd.DataFrame({
            "ticker": ["000001.SZ", "000002.SZ", "000003.SZ", "0700.HK"],
            "snapshot_date": ["2023-06-30", "2024-06-30", "2025-06-30", "2025-06-30"],
            "close": [2.0, 3.0, 4.0, 5.0],
            "market_cap_rmb": [1_000_000_000.0, 1_000_000_000.0, 1_000_000_000.0, 1_000_000_000.0],
            "price_return_5d": [0.2, 0.3, 0.4, 0.5],
        })
        incumbent_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {"backend": "incumbent"},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        candidate_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["price_return_5d"],
            "ticker_priors": {},
            "metadata": {"backend": "candidate"},
            "models": [{
                "clip_low": [-1.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [100.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        recent_payload = _conditional_blend_payload(
            anchor_payload=incumbent_payload,
            candidate_payload=candidate_payload,
            routes=[{
                "market": "ashare",
                "cap_bucket": "small",
                "candidate_weight": 1.0,
            }],
        )
        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": [],
            "ticker_priors": {},
            "metadata": {
                "backend": "temporal_recent_segment_regime_blend",
                "temporal_recent_regime_start": "2025-01-01",
            },
            "temporal_payload_members": [
                {"end_date": "2023-07-07", "payload": candidate_payload},
                {
                    "start_date": "2023-07-07",
                    "end_date": "2025-01-01",
                    "payload": incumbent_payload,
                },
                {"start_date": "2025-01-01", "payload": recent_payload},
            ],
        }

        compacted = _compact_runtime_payload(payload)

        assert isinstance(compacted, dict)
        assert "conditional_blend" in compacted
        assert "temporal_payload_members" not in compacted
        assert predict_ml_ranker_payload(frame, compacted).tolist() == pytest.approx(
            predict_ml_ranker_payload(frame, payload).tolist()
        )
        assert predict_ml_ranker_payload(frame, compacted).tolist() == pytest.approx(
            [20.0, 3.0, 40.0, 5.0]
        )

    def test_recent_market_regime_temporal_payload_routes_by_date_and_market(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _recent_market_regime_temporal_payload,
            predict_ml_ranker_payload,
        )

        frame = pd.DataFrame({
            "ticker": ["000001.SZ", "000002.SZ", "000003.SZ", "0700.HK"],
            "snapshot_date": ["2023-06-30", "2024-06-30", "2025-06-30", "2025-06-30"],
            "close": [2.0, 3.0, 4.0, 5.0],
            "price_return_5d": [0.2, 0.3, 0.4, 0.5],
        })
        incumbent_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["close"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [0.0],
                "clip_high": [100.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        candidate_payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["price_return_5d"],
            "ticker_priors": {},
            "metadata": {},
            "models": [{
                "clip_low": [-1.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [100.0],
                "intercept": 0.0,
                "weight": 1.0,
            }],
        }
        payload = _recent_market_regime_temporal_payload(
            candidate_payload=candidate_payload,
            incumbent_payload=incumbent_payload,
            gate_start="2023-07-07",
            recent_start="2025-01-01",
            candidate_weights_by_market={"ashare": 1.0},
        )

        assert "conditional_blend" in payload
        assert "temporal_payload_members" not in payload
        predictions = predict_ml_ranker_payload(frame, payload)

        assert predictions.tolist() == pytest.approx([20.0, 3.0, 40.0, 5.0])

    def test_ml_trainer_candidate_choice_normalization(self) -> None:
        from valueinvestor.scorer_improver import ml_trainer

        choices = ml_trainer._normalize_candidate_choices(
            ("core,short-horizon",),
            aliases=ml_trainer.FEATURE_SET_ALIASES,
            option_name="candidate_feature_sets",
        )

        assert choices == ("core", "short_horizon")

        with pytest.raises(ValueError, match="cannot mix auto/all"):
            ml_trainer._normalize_candidate_choices(
                ("auto", "core"),
                aliases=ml_trainer.FEATURE_SET_ALIASES,
                option_name="candidate_feature_sets",
            )

        targets = ml_trainer._normalize_candidate_targets(
            ("target-rank-1w,target-rank-1w-market",),
            allowed_targets=("target_rank_1w", "target_rank_1w_market"),
        )
        six_month_targets = ml_trainer._normalize_candidate_targets(
            ("rank-6m-soft,weighted-802,weighted-8515,weighted-703-market,mean",),
            allowed_targets=(
                "target_rank_6m_soft",
                "target_rank_weighted_802",
                "target_rank_weighted_8515",
                "target_rank_weighted_703_market",
                "target_rank_mean",
            ),
        )
        lambdas = ml_trainer._normalize_candidate_lambdas(("1,3,10",))

        assert targets == ("target_rank_1w", "target_rank_1w_market")
        assert six_month_targets == (
            "target_rank_6m_soft",
            "target_rank_weighted_802",
            "target_rank_weighted_8515",
            "target_rank_weighted_703_market",
            "target_rank_mean",
        )
        assert lambdas == (1.0, 3.0, 10.0)

    def test_recency_sample_weights_prioritize_newer_snapshots(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _recency_sample_weights

        snapshots = pd.DataFrame({
            "snapshot_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        })

        weights = _recency_sample_weights(snapshots, half_life_days=365.0)

        assert weights[2] > weights[1] > weights[0]
        assert weights.mean() == pytest.approx(1.0)

    def test_sample_training_snapshots_uses_seeded_balanced_sampling(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _sample_training_snapshots

        rows = [
            {"snapshot_date": snapshot_date, "ticker": f"T{row_idx:03d}"}
            for snapshot_date in pd.date_range("2020-01-01", periods=3, freq="MS")
            for row_idx in range(20)
        ]
        snapshots = pd.DataFrame(rows)

        first = _sample_training_snapshots(snapshots, 30, random_seed=0)
        repeat = _sample_training_snapshots(snapshots, 30, random_seed=0)
        alternate = _sample_training_snapshots(snapshots, 30, random_seed=41)

        assert first["ticker"].tolist() == repeat["ticker"].tolist()
        assert first["ticker"].tolist() != alternate["ticker"].tolist()
        assert first.groupby("snapshot_date").size().tolist() == [10, 10, 10]

    def test_build_full_eval_ensembles_averages_compatible_candidates(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _build_full_eval_ensembles

        rows = []
        predictions_a = []
        predictions_b = []
        for snapshot_date in pd.date_range("2020-01-01", periods=2, freq="MS"):
            for rank in range(10):
                rows.append({
                    "snapshot_date": snapshot_date,
                    "ticker": f"T{rank:03d}",
                    "forward_return_1m": float(rank),
                    "forward_return_3m": float(rank),
                    "forward_return_6m": float(rank),
                })
                predictions_a.append(float(rank))
                predictions_b.append(float(rank) + (0.1 if rank % 2 else -0.1))
        snapshots = pd.DataFrame(rows)

        def candidate(name: str, predictions: list[float], rho: float) -> dict:
            return {
                "target": name,
                "ridge_lambda": 100.0,
                "backend": "numpy",
                "model_kind": "ridge",
                "feature_set": "core",
                "feature_names": ["x"],
                "prior_strategy": "no_ticker_priors",
                "prefer_row_priors": False,
                "priors": {},
                "clip_low": [0.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "models": None,
                "metrics": {"6m": {"spearman_rho": rho}},
                "sample_rows": len(snapshots),
                "sample_seed": 0,
                "_full_predictions": predictions,
            }

        ensembles = _build_full_eval_ensembles(
            [candidate("a", predictions_a, 0.90), candidate("b", predictions_b, 0.89)],
            snapshots,
            train_baseline_rho=0.1,
            primary_horizon="6m",
        )

        assert len(ensembles) >= 3
        assert all(ensemble["target"] == "ensemble" for ensemble in ensembles)
        assert all(len(ensemble["ensemble_members"]) == 2 for ensemble in ensembles)
        assert any(
            ensemble["metrics"]["6m"]["spearman_rho"] == pytest.approx(1.0)
            for ensemble in ensembles
        )
        assert any(
            any(model["weight"] != 1.0 for model in ensemble["models"])
            for ensemble in ensembles
        )

    def test_build_full_eval_ensembles_keeps_rolling_prior_flag(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _build_full_eval_ensembles

        snapshots = pd.DataFrame(
            [
                {
                    "snapshot_date": snapshot_date,
                    "ticker": f"T{rank:03d}",
                    "forward_return_1m": float(rank),
                    "forward_return_3m": float(rank),
                    "forward_return_6m": float(rank),
                }
                for snapshot_date in pd.date_range("2020-01-01", periods=2, freq="MS")
                for rank in range(10)
            ]
        )
        predictions = [float(i % 10) for i in range(len(snapshots))]

        def candidate(name: str, rho: float) -> dict:
            return {
                "target": name,
                "ridge_lambda": 3.0,
                "backend": "mlx",
                "model_kind": "market_ridge_recent_365",
                "feature_set": "short_horizon",
                "feature_names": ["x"],
                "prior_strategy": "rolling_ticker_priors",
                "prefer_row_priors": True,
                "priors": {},
                "clip_low": [0.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "models": None,
                "metrics": {"6m": {"spearman_rho": rho}},
                "sample_rows": len(snapshots),
                "sample_seed": 0,
                "_full_predictions": predictions,
            }

        ensembles = _build_full_eval_ensembles(
            [candidate("a", 0.90), candidate("b", 0.89)],
            snapshots,
            train_baseline_rho=0.1,
            primary_horizon="6m",
        )

        assert ensembles
        assert all(ensemble["prefer_row_priors"] for ensemble in ensembles)
        assert all(
            ensemble["prior_strategy"] == "rolling_ticker_priors"
            for ensemble in ensembles
        )

    def test_temporal_anchor_candidates_interleave_gate_scored_anchors(self) -> None:
        import numpy as np

        from valueinvestor.scorer_improver.ml_trainer import (
            _build_temporal_anchor_candidates,
        )

        def candidate(name: str, rho: float) -> dict:
            return {
                "target": name,
                "ridge_lambda": 1.0,
                "backend": "mlx",
                "model_kind": "market_ridge_recent_1460",
                "feature_set": "short_horizon",
                "feature_names": ["x"],
                "prior_strategy": "ticker_priors",
                "prefer_row_priors": False,
                "priors": {},
                "metrics": {"6m": {"spearman_rho": rho}},
                "sample_metrics": {"6m": {"spearman_rho": rho}},
                "sample_improvement": 1.0,
                "sample_rows": 10,
                "sample_seed": 0,
                "_full_predictions": np.asarray([rho, rho + 0.1], dtype="float32"),
            }

        temporal = _build_temporal_anchor_candidates(
            [candidate("train_best", 0.31), candidate("train_second", 0.30)],
            train_baseline_rho=0.10,
            primary_horizon="6m",
            gate_start="2023-07-07",
            anchor_payloads=[
                {"path": "weak.json", "payload": {"metadata": {}}, "gate_rho": 0.18},
                {"path": "strong.json", "payload": {"metadata": {}}, "gate_rho": 0.26},
            ],
            promotion_min_train_rho=0.25,
            max_candidates=3,
        )

        assert [candidate["temporal_anchor_path"] for candidate in temporal] == [
            "strong.json",
            "strong.json",
            "weak.json",
        ]
        assert temporal[0]["temporal_anchor_gate_rho"] == pytest.approx(0.26)
        assert temporal[0]["metrics"]["6m"]["spearman_rho"] == pytest.approx(0.31)

    def test_recent_market_ridge_model_kind_helpers(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _fit_model_kind_for_candidate,
            _recent_half_life_for_model_kind,
            _six_month_gate_route_raw_degradation_limit,
            _six_month_recent_segment_route_max_candidates,
            _six_month_recent_segment_route_max_segments,
            _six_month_recent_segment_route_max_starts,
            _six_month_recent_segment_route_weights,
            _six_month_recency_half_lives_days,
            _six_month_sample_seeds,
        )

        assert _fit_model_kind_for_candidate("market_ridge_recent_270") == "market_ridge"
        assert _fit_model_kind_for_candidate("segment_ridge_recent_270") == "segment_ridge"
        assert _recent_half_life_for_model_kind("market_ridge_recent_270") == 270.0
        assert _recent_half_life_for_model_kind("segment_ridge_recent_270") == 270.0
        assert _recent_half_life_for_model_kind("market_ridge") is None
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS", raising=False)
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT", raising=False)
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS", raising=False)
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_STARTS", raising=False)
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS", raising=False)
            monkeypatch.delenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES", raising=False)
            assert 2190.0 in _six_month_recency_half_lives_days()
            assert _six_month_gate_route_raw_degradation_limit() == pytest.approx(0.005)
            assert _six_month_recent_segment_route_weights() == (1.0, 0.75, 0.5)
            assert _six_month_recent_segment_route_max_starts() == 4
            assert _six_month_recent_segment_route_max_segments() == 8
            assert _six_month_recent_segment_route_max_candidates() == 8
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_RECENCY_HALF_LIVES_DAYS", "1095,1460")
            assert _six_month_recency_half_lives_days() == (1095.0, 1460.0)
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_GATE_ROUTE_RAW_DEGRADATION_LIMIT", "0.012")
            assert _six_month_gate_route_raw_degradation_limit() == pytest.approx(0.012)
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_WEIGHTS", "1,0.5,1")
            assert _six_month_recent_segment_route_weights() == (1.0, 0.5)
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_STARTS", "3")
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_SEGMENTS", "5")
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_RECENT_SEGMENT_ROUTE_MAX_CANDIDATES", "7")
            assert _six_month_recent_segment_route_max_starts() == 3
            assert _six_month_recent_segment_route_max_segments() == 5
            assert _six_month_recent_segment_route_max_candidates() == 7
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_SAMPLE_SEEDS", "0,29,0")
            assert _six_month_sample_seeds() == (0, 29)

    def test_market_blend_helpers_apply_market_specific_weights(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _market_blend_weight_vector,
            _market_weighted_blend_payload,
        )

        def model(market: str) -> dict[str, object]:
            return {
                "clip_low": [0.0],
                "clip_high": [1.0],
                "mean": [0.0],
                "scale": [1.0],
                "coef": [1.0],
                "intercept": 0.0,
                "market": market,
            }

        payload = {
            "schema_version": "ml-ranker-v1",
            "feature_names": ["x"],
            "ticker_priors": {},
            "metadata": {},
            "models": [model("ashare"), model("hk")],
        }

        weights = {"ashare": 0.1, "hk": 0.5}
        vector = _market_blend_weight_vector(
            pd.DataFrame({"ticker": ["000001.SZ", "00700.HK"]}),
            weights,
        )
        blended = _market_weighted_blend_payload(
            payload,
            payload,
            candidate_weights_by_market=weights,
        )

        assert vector.tolist() == pytest.approx([0.1, 0.5])
        assert [model["weight"] for model in blended["models"]] == pytest.approx(
            [0.9, 0.5, 0.1, 0.5]
        )

    def test_train_floor_blend_selection_keeps_near_floor_and_strong_blends(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _six_month_gate_market_blend_limit,
            _select_train_floor_blends,
            _select_train_floor_market_blends,
            _top_gate_anchor_payloads,
        )

        weights = _select_train_floor_blends(
            [(0.05, 0.230), (0.10, 0.239), (0.125, 0.242), (0.20, 0.255)],
            promotion_min_train_rho=0.241,
            limit=2,
        )
        market_weights = _select_train_floor_market_blends(
            [
                ({"ashare": 0.0, "hk": 0.25}, 0.240),
                ({"ashare": 0.0, "hk": 0.35}, 0.244),
                ({"ashare": 0.025, "hk": 0.25}, 0.242),
                ({"ashare": 0.05, "hk": 0.50}, 0.260),
            ],
            promotion_min_train_rho=0.241,
            limit=2,
        )

        assert weights == pytest.approx((0.125, 0.20))
        assert market_weights == (
            {"ashare": 0.025, "hk": 0.25},
            {"ashare": 0.0, "hk": 0.35},
        )

        low_impact_weights = _select_train_floor_blends(
            [
                (0.01, 0.240),
                (0.02, 0.2411),
                (0.03, 0.2412),
                (0.05, 0.2420),
                (1.00, 0.2600),
            ],
            promotion_min_train_rho=0.241,
            limit=3,
        )
        assert low_impact_weights == pytest.approx((0.02, 0.03, 0.05))

        low_impact_market_weights = _select_train_floor_market_blends(
            [
                ({"ashare": 0.005, "hk": 0.0}, 0.2409),
                ({"ashare": 0.010, "hk": 0.0}, 0.2411),
                ({"ashare": 0.015, "hk": 0.0}, 0.2412),
                ({"ashare": 0.050, "hk": 0.0}, 0.2420),
                ({"ashare": 0.050, "hk": 0.5}, 0.2600),
            ],
            promotion_min_train_rho=0.241,
            limit=3,
        )
        assert low_impact_market_weights == (
            {"ashare": 0.010, "hk": 0.0},
            {"ashare": 0.015, "hk": 0.0},
            {"ashare": 0.050, "hk": 0.0},
        )

        tolerant_market_weights = _select_train_floor_market_blends(
            [
                ({"ashare": 0.005, "hk": 0.0}, 0.2409),
                ({"ashare": 0.010, "hk": 0.0}, 0.2411),
            ],
            promotion_min_train_rho=0.241,
            limit=2,
            train_floor_tolerance=0.0002,
        )
        assert tolerant_market_weights == (
            {"ashare": 0.005, "hk": 0.0},
            {"ashare": 0.010, "hk": 0.0},
        )

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT", "9")
            assert _six_month_gate_market_blend_limit() == 9
            monkeypatch.setenv("VALUEINVESTOR_ML_6M_GATE_MARKET_BLEND_LIMIT", "bad")
            assert _six_month_gate_market_blend_limit() == 6

        top_anchors = _top_gate_anchor_payloads(
            [
                {"path": "weak", "gate_rho": 0.10},
                {"path": "best", "gate_rho": 0.30},
                {"path": "second", "gate_rho": 0.20},
            ],
            limit=2,
        )
        assert [anchor["path"] for anchor in top_anchors] == ["best", "second"]

    def test_full_eval_candidate_selection_keeps_strategy_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        candidates = [
            {
                "name": "ticker_best",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.10}},
            },
            {
                "name": "ticker_second",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.09}},
            },
            {
                "name": "no_prior",
                "prior_strategy": "no_ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.02}},
            },
        ]

        selected = _select_full_eval_candidates(candidates, limit=2, primary_horizon="1w")

        assert [candidate["name"] for candidate in selected] == ["no_prior", "ticker_best"]

    def test_full_eval_candidate_selection_keeps_best_and_current_6m_family(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        candidates = [
            {
                "name": "market_short_rank",
                "backend": "mlx",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge",
                "target": "target_rank_6m",
                "ridge_lambda": 10.0,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": 0.11}},
            },
            {
                "name": "sample_best_mean",
                "backend": "numpy",
                "feature_set": "core",
                "model_kind": "ridge",
                "target": "target_rank_mean",
                "ridge_lambda": 100.0,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": 0.12}},
            },
            {
                "name": "soft_3000",
                "backend": "mlx",
                "feature_set": "core",
                "model_kind": "ridge",
                "target": "target_rank_6m_soft",
                "ridge_lambda": 3_000.0,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": 0.10}},
            },
            {
                "name": "soft_no_priors",
                "backend": "mlx",
                "feature_set": "core",
                "model_kind": "ridge",
                "target": "target_rank_6m_soft",
                "ridge_lambda": 1_000.0,
                "prior_strategy": "no_ticker_priors",
                "metrics": {"6m": {"spearman_rho": 0.10}},
            },
            {
                "name": "weighted_802",
                "backend": "mlx",
                "feature_set": "core",
                "model_kind": "ridge",
                "target": "target_rank_weighted_802",
                "ridge_lambda": 300.0,
                "prior_strategy": "rolling_ticker_priors",
                "metrics": {"6m": {"spearman_rho": 0.09}},
            },
        ]

        selected = _select_full_eval_candidates(candidates, limit=2, primary_horizon="6m")

        assert [candidate["name"] for candidate in selected] == [
            "sample_best_mean",
            "market_short_rank",
        ]

    def test_full_eval_candidate_selection_uses_6m_walk_forward_after_validation(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        def candidate(name: str, rho: float, wf_delta: float) -> dict:
            return {
                "name": name,
                "backend": "mlx",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge",
                "target": "target_rank_6m",
                "ridge_lambda": 10.0,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": rho}},
                "walk_forward": {
                    "mean_deltas": {"6m": wf_delta},
                    "median_deltas": {"6m": wf_delta / 2.0},
                },
            }

        selected = _select_full_eval_candidates(
            [
                candidate("sample_best", 0.40, 0.001),
                candidate("walk_forward_best", 0.35, 0.010),
            ],
            limit=1,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == ["walk_forward_best"]

    def test_full_eval_candidate_selection_uses_6m_gate_probe_when_available(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        def candidate(name: str, train_rho: float, gate_rho: float) -> dict:
            return {
                "name": name,
                "backend": "mlx",
                "feature_set": "cross_sectional_interactions",
                "model_kind": "segment_ridge_recent_2190",
                "target": "target_rank_weighted_901_market",
                "ridge_lambda": 0.1,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": train_rho}},
                "gate_probe": {
                    "metrics": {"6m": {"spearman_rho": gate_rho}},
                    "rows": 120,
                },
            }

        selected = _select_full_eval_candidates(
            [
                candidate("sample_best_gate_bad", 0.41, 0.08),
                candidate("sample_second_gate_good", 0.38, 0.25),
            ],
            limit=1,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == ["sample_second_gate_good"]

    def test_full_eval_candidate_selection_keeps_6m_model_kind_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        def candidate(name: str, rho: float, model_kind: str) -> dict:
            return {
                "name": name,
                "backend": "mlx",
                "feature_set": "short_horizon",
                "model_kind": model_kind,
                "target": "target_rank_weighted_802",
                "ridge_lambda": 3.0,
                "prior_strategy": "ticker_priors",
                "metrics": {"6m": {"spearman_rho": rho}},
            }

        selected = _select_full_eval_candidates(
            [
                candidate("market_best", 0.40, "market_ridge"),
                candidate("market_second", 0.39, "market_ridge"),
                candidate("recent", 0.30, "market_ridge_recent"),
                candidate("ridge", 0.29, "ridge"),
            ],
            limit=3,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "market_best",
            "recent",
            "ridge",
        ]

    def test_full_eval_candidate_selection_keeps_6m_prior_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        def candidate(name: str, rho: float, prior_strategy: str) -> dict:
            return {
                "name": name,
                "backend": "mlx",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge",
                "target": "target_rank_weighted_703",
                "ridge_lambda": 3.0,
                "prior_strategy": prior_strategy,
                "metrics": {"6m": {"spearman_rho": rho}},
            }

        selected = _select_full_eval_candidates(
            [
                candidate("ticker_best", 0.40, "ticker_priors"),
                candidate("ticker_second", 0.39, "ticker_priors"),
                candidate("no_ticker", 0.30, "no_ticker_priors"),
            ],
            limit=2,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "ticker_best",
            "no_ticker",
        ]

    def test_walk_forward_shortlist_respects_full_eval_limit(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _walk_forward_candidate_limit

        assert _walk_forward_candidate_limit(4, primary_horizon="6m") == 4
        assert _walk_forward_candidate_limit(12, primary_horizon="6m") == 12
        assert _walk_forward_candidate_limit(4, primary_horizon="1w") == 4

    def test_promotion_gate_candidate_selection_uses_walk_forward_and_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(
            name: str,
            rho: float,
            *,
            target: str = "target_rank_6m",
            model_kind: str = "market_ridge",
            wf_delta: float | None = None,
        ) -> dict:
            result = {
                "name": name,
                "target": target,
                "model_kind": model_kind,
                "ridge_lambda": 10.0,
                "metrics": {"6m": {"spearman_rho": rho}},
            }
            if wf_delta is not None:
                result["walk_forward"] = {
                    "mean_deltas": {"6m": wf_delta},
                    "median_deltas": {"6m": wf_delta / 2.0},
                }
            return result

        selected = _select_promotion_gate_candidates(
            [
                candidate("train_best", 0.40, wf_delta=0.001),
                candidate("walk_forward_best", 0.35, target="target_rank_weighted_703", wf_delta=0.01),
                candidate("ensemble", 0.39, target="ensemble", model_kind="ensemble"),
                candidate("duplicate_family", 0.38, wf_delta=0.002),
            ],
            limit=3,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "train_best",
            "ensemble",
            "walk_forward_best",
        ]

    def test_promotion_gate_candidate_selection_uses_6m_gate_probe_when_available(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(name: str, train_rho: float, gate_rho: float | None) -> dict:
            result = {
                "name": name,
                "target": "target_rank_weighted_901",
                "model_kind": "segment_ridge_recent",
                "ridge_lambda": 0.3,
                "metrics": {"6m": {"spearman_rho": train_rho}},
            }
            if gate_rho is not None:
                result["gate_probe"] = {
                    "metrics": {"6m": {"spearman_rho": gate_rho}},
                    "rows": 120,
                }
            return result

        selected = _select_promotion_gate_candidates(
            [
                candidate("train_best_gate_bad", 0.41, 0.08),
                candidate("probe_best", 0.39, 0.12),
                candidate("ensemble_without_probe", 0.40, None),
            ],
            limit=1,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == ["probe_best"]

    def test_promotion_gate_candidate_selection_keeps_temporal_anchor_with_gate_probe(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(
            name: str,
            train_rho: float,
            gate_rho: float | None,
            *,
            temporal_anchor_gate_rho: float | None = None,
        ) -> dict:
            result = {
                "name": name,
                "target": "target_rank_weighted_901",
                "model_kind": "segment_ridge_recent",
                "ridge_lambda": 0.3,
                "metrics": {"6m": {"spearman_rho": train_rho}},
            }
            if gate_rho is not None:
                result["gate_probe"] = {
                    "metrics": {"6m": {"spearman_rho": gate_rho}},
                    "rows": 120,
                }
            if temporal_anchor_gate_rho is not None:
                result.update({
                    "target": "temporal_anchor",
                    "target_source": "temporal_regime",
                    "temporal_anchor_gate_rho": temporal_anchor_gate_rho,
                })
            return result

        selected = _select_promotion_gate_candidates(
            [
                candidate("probe_best", 0.39, 0.12),
                candidate(
                    "temporal_anchor",
                    0.38,
                    None,
                    temporal_anchor_gate_rho=0.18,
                ),
            ],
            limit=1,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == ["temporal_anchor"]

    def test_promotion_gate_candidate_selection_interleaves_walk_forward_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(
            name: str,
            rho: float,
            *,
            target: str = "target_rank_6m",
            wf_delta: float = 0.0,
        ) -> dict:
            return {
                "name": name,
                "target": target,
                "model_kind": "market_ridge",
                "ridge_lambda": 10.0,
                "metrics": {"6m": {"spearman_rho": rho}},
                "walk_forward": {
                    "mean_deltas": {"6m": wf_delta},
                    "median_deltas": {"6m": wf_delta},
                },
            }

        selected = _select_promotion_gate_candidates(
            [
                candidate("train_best", 0.40, target="target_rank_6m", wf_delta=0.001),
                candidate("train_second_same_family", 0.39, target="target_rank_6m", wf_delta=0.002),
                candidate("walk_forward_diverse", 0.35, target="target_rank_weighted_8515", wf_delta=0.020),
                candidate("walk_forward_next", 0.34, target="target_rank_weighted_901", wf_delta=0.010),
            ],
            limit=3,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "train_best",
            "walk_forward_diverse",
            "walk_forward_next",
        ]

    def test_promotion_gate_candidate_selection_does_not_only_select_temporal(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(
            name: str,
            rho: float,
            *,
            target: str,
            target_source: str = "direct",
        ) -> dict:
            return {
                "name": name,
                "target": target,
                "target_source": target_source,
                "model_kind": "temporal_payload_blend"
                if target_source == "temporal_regime"
                else "market_ridge",
                "ridge_lambda": 0.0 if target_source == "temporal_regime" else 3.0,
                "prior_strategy": "temporal"
                if target_source == "temporal_regime"
                else "ticker_priors",
                "metrics": {"6m": {"spearman_rho": rho}},
            }

        selected = _select_promotion_gate_candidates(
            [
                candidate(
                    "temporal_best",
                    0.42,
                    target="temporal_anchor",
                    target_source="temporal_regime",
                ),
                candidate(
                    "temporal_second",
                    0.41,
                    target="temporal_anchor",
                    target_source="temporal_regime",
                ),
                candidate("raw_best", 0.40, target="target_rank_weighted_901"),
                candidate("raw_diverse", 0.39, target="target_rank_6m_market"),
            ],
            limit=3,
            primary_horizon="6m",
        )

        assert "temporal_best" in [candidate["name"] for candidate in selected]
        assert any(
            candidate.get("target_source") != "temporal_regime" for candidate in selected
        )

    def test_promotion_gate_candidate_selection_keeps_6m_model_kind_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(name: str, rho: float, model_kind: str) -> dict:
            return {
                "name": name,
                "target": "target_rank_weighted_802",
                "model_kind": model_kind,
                "ridge_lambda": 3.0,
                "metrics": {"6m": {"spearman_rho": rho}},
            }

        selected = _select_promotion_gate_candidates(
            [
                candidate("market_best", 0.40, "market_ridge"),
                candidate("market_second", 0.39, "market_ridge"),
                candidate("recent", 0.30, "market_ridge_recent"),
                candidate("ridge", 0.29, "ridge"),
            ],
            limit=3,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "market_best",
            "recent",
            "ridge",
        ]

    def test_promotion_gate_candidate_selection_keeps_6m_prior_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_promotion_gate_candidates

        def candidate(name: str, rho: float, prior_strategy: str) -> dict:
            return {
                "name": name,
                "target": "target_rank_weighted_703",
                "model_kind": "market_ridge",
                "ridge_lambda": 3.0,
                "prior_strategy": prior_strategy,
                "metrics": {"6m": {"spearman_rho": rho}},
            }

        selected = _select_promotion_gate_candidates(
            [
                candidate("ticker_best", 0.40, "ticker_priors"),
                candidate("ticker_second", 0.39, "ticker_priors"),
                candidate("no_ticker", 0.30, "no_ticker_priors"),
            ],
            limit=2,
            primary_horizon="6m",
        )

        assert [candidate["name"] for candidate in selected] == [
            "ticker_best",
            "no_ticker",
        ]

    def test_ml_trainer_reads_incumbent_train_rho_from_payload(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _payload_train_rho

        payload = {
            "metadata": {
                "train_metrics": {
                    "6m": {"spearman_rho": 0.310611},
                },
            },
        }

        assert _payload_train_rho(payload, "6m") == pytest.approx(0.310611)
        assert _payload_train_rho(payload, "1w") is None
        assert _payload_train_rho(None, "6m") is None

    def test_primary_spearman_helper_matches_full_evaluation(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            _evaluate_prepared_primary_spearman_rho,
            _evaluate_predictions,
            _evaluate_primary_spearman_rho,
            _prepare_primary_spearman_context,
        )

        rows = []
        predictions = []
        for snap_idx, snap_date in enumerate(pd.date_range("2024-01-01", periods=3, freq="D")):
            for stock_idx in range(12):
                rows.append(
                    {
                        "snapshot_date": snap_date.date(),
                        "forward_return_6m": stock_idx + snap_idx * 0.01,
                    }
                )
                predictions.append(float(11 - stock_idx if snap_idx == 1 else stock_idx))
        frame = pd.DataFrame(rows)
        prediction_array = pd.Series(predictions).to_numpy(dtype="float64", copy=True)

        full_metrics = _evaluate_predictions(frame, prediction_array)
        primary_rho = _evaluate_primary_spearman_rho(frame, prediction_array, "6m")
        context = _prepare_primary_spearman_context(frame, "6m")
        prepared_rho = _evaluate_prepared_primary_spearman_rho(context, prediction_array)

        assert primary_rho == pytest.approx(full_metrics["6m"]["spearman_rho"])
        assert prepared_rho == pytest.approx(primary_rho)

        prediction_array[0] = float("nan")
        primary_rho = _evaluate_primary_spearman_rho(frame, prediction_array, "6m")
        prepared_rho = _evaluate_prepared_primary_spearman_rho(context, prediction_array)

        assert prepared_rho == pytest.approx(primary_rho)

    def test_routed_model_preselection_matches_runtime_fallback(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import (
            build_feature_matrix,
            _predict_scaled_linear_models,
            _runtime_feature_parity,
        )
        from valueinvestor.screener.ml_ranker import (
            MODEL_SCHEMA_VERSION,
            expected_feature_names,
        )

        snapshots = pd.DataFrame(
            {
                "ticker": ["000001", "0700.HK", "000002"],
                "snapshot_date": [pd.Timestamp("2025-01-31").date()] * 3,
                "market_cap_rmb": [6e9, 60e9, 20e9],
                "pe_ratio": [1.0, 2.0, 3.0],
            }
        )
        feature_names = expected_feature_names()
        n_features = len(feature_names)
        normalizer = {
            "clip_low": [-1e12] * n_features,
            "clip_high": [1e12] * n_features,
            "mean": [0.0] * n_features,
            "scale": [1.0] * n_features,
        }
        first_coef = [0.0] * n_features
        first_coef[0] = 1.0
        second_coef = [0.0] * n_features
        second_coef[0] = 2.0
        models = [
            {
                **normalizer,
                "coef": first_coef,
                "intercept": 0.0,
                "market": "ashare",
                "cap_bucket": "small",
            },
            {
                **normalizer,
                "coef": second_coef,
                "intercept": 0.0,
                "market": "hk",
                "cap_bucket": "large",
            },
        ]

        feature_matrix = build_feature_matrix(
            snapshots,
            {},
            feature_names=feature_names,
        )
        preselection = _predict_scaled_linear_models(
            feature_matrix,
            snapshots,
            models,
        )
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": feature_names,
            "models": models,
            "ticker_priors": {},
            "metadata": {"target_horizon": "6m"},
        }
        parity = _runtime_feature_parity(snapshots, payload)

        expected = [
            feature_matrix[0, 0],
            2.0 * feature_matrix[1, 0],
            1.5 * feature_matrix[2, 0],
        ]
        assert preselection.tolist() == pytest.approx(expected)
        assert parity["matched"] is True
        assert parity["prediction_max_abs_error"] == pytest.approx(0.0)

    def test_strict_walk_forward_panel_balances_configuration_axes(self) -> None:
        from itertools import product

        from valueinvestor.scorer_improver.ml_trainer import (
            _select_balanced_walk_forward_specs,
        )

        fields = (
            "model_kind",
            "prior_strategy",
            "feature_set",
            "target",
            "ridge_lambda",
        )
        values = (
            ("ridge", "market_ridge", "segment_ridge"),
            ("rolling_ticker_priors", "no_ticker_priors"),
            ("cross_sectional", "cross_sectional_interactions"),
            ("target_rank_6m", "target_rank_6m_market", "target_rank_6m_soft"),
            (30.0, 1_000.0),
        )
        candidates = [dict(zip(fields, combination)) for combination in product(*values)]

        selected = _select_balanced_walk_forward_specs(candidates, limit=12)

        assert len(selected) == 12
        for field, expected_values in zip(fields, values):
            assert {candidate[field] for candidate in selected} == set(expected_values)

    def test_full_eval_candidate_selection_prioritizes_current_1w_winning_family(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        candidates = [
            {
                "name": "sample_best_market",
                "prior_strategy": "no_ticker_priors",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.12}},
            },
            {
                "name": "recent_target_1w",
                "prior_strategy": "no_ticker_priors",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge_recent",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.10}},
            },
            {
                "name": "target_market",
                "prior_strategy": "no_ticker_priors",
                "feature_set": "short_horizon",
                "model_kind": "market_ridge",
                "target": "target_rank_1w_market",
                "metrics": {"1w": {"spearman_rho": 0.11}},
            },
        ]

        selected = _select_full_eval_candidates(candidates, limit=2, primary_horizon="1w")

        assert [candidate["name"] for candidate in selected] == [
            "recent_target_1w",
            "sample_best_market",
        ]

    def test_full_eval_candidate_selection_keeps_backend_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        candidates = [
            {
                "name": "cg_best",
                "backend": "mlx",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.10}},
            },
            {
                "name": "cg_second",
                "backend": "mlx",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.09}},
            },
            {
                "name": "adam",
                "backend": "mlx-adam",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.08}},
            },
        ]

        selected = _select_full_eval_candidates(candidates, limit=2, primary_horizon="1w")

        assert [candidate["name"] for candidate in selected] == ["cg_best", "adam"]

    def test_full_eval_candidate_selection_keeps_prior_backend_pair_diversity(self) -> None:
        from valueinvestor.scorer_improver.ml_trainer import _select_full_eval_candidates

        candidates = [
            {
                "name": "ticker_cg",
                "backend": "mlx",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.10}},
            },
            {
                "name": "ticker_adam",
                "backend": "mlx-adam",
                "prior_strategy": "ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.09}},
            },
            {
                "name": "no_prior_cg",
                "backend": "mlx",
                "prior_strategy": "no_ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.02}},
            },
            {
                "name": "no_prior_adam",
                "backend": "mlx-adam",
                "prior_strategy": "no_ticker_priors",
                "model_kind": "ridge",
                "target": "target_rank_1w",
                "metrics": {"1w": {"spearman_rho": 0.01}},
            },
        ]

        selected = _select_full_eval_candidates(candidates, limit=4, primary_horizon="1w")

        assert [candidate["name"] for candidate in selected] == [
            "no_prior_cg",
            "ticker_cg",
            "ticker_adam",
            "no_prior_adam",
        ]

    def test_gate_result_selection_prefers_higher_primary_delta(self) -> None:
        from types import SimpleNamespace

        from valueinvestor.scorer_improver.ml_trainer import _is_better_gate_result

        incumbent = SimpleNamespace(deltas={"1w": 0.0001}, weighted_utility=0.01)
        higher_delta = SimpleNamespace(deltas={"1w": 0.0002}, weighted_utility=-0.01)
        lower_delta = SimpleNamespace(deltas={"1w": 0.00005}, weighted_utility=0.02)
        same_delta_higher_utility = SimpleNamespace(deltas={"1w": 0.0001}, weighted_utility=0.02)

        assert _is_better_gate_result(higher_delta, incumbent, primary_horizon="1w")
        assert not _is_better_gate_result(lower_delta, incumbent, primary_horizon="1w")
        assert _is_better_gate_result(
            same_delta_higher_utility,
            incumbent,
            primary_horizon="1w",
        )

    def test_pairwise_ranker_learns_positive_ordering(self) -> None:
        import numpy as np

        from valueinvestor.scorer_improver.ml_trainer import _fit_pairwise_ranker_numpy

        X = np.asarray([[0.0], [1.0]] * 10)
        y = np.asarray([-1.0, 1.0] * 10)
        snapshot_dates = pd.Series(
            pd.date_range("2020-01-01", periods=10, freq="MS").repeat(2)
        )

        coef, backend = _fit_pairwise_ranker_numpy(
            X,
            y,
            snapshot_dates,
            ridge_lambda=0.1,
            max_pairs=100,
            steps=30,
        )

        assert backend == "pairwise_numpy"
        assert coef[0] > 0

    def test_mlx_ridge_matches_numpy_when_available(self) -> None:
        import numpy as np

        pytest.importorskip("mlx.core")
        from valueinvestor.scorer_improver.ml_trainer import (
            _fit_ridge_mlx,
            _fit_ridge_numpy,
        )

        rng = np.random.default_rng(11)
        X = rng.normal(size=(256, 12)).astype("float64")
        y = rng.normal(size=256).astype("float64")

        mlx_coef, backend = _fit_ridge_mlx(X, y, 10.0)
        numpy_coef, _ = _fit_ridge_numpy(X, y, 10.0)

        assert backend == "mlx"
        assert np.max(np.abs(mlx_coef - numpy_coef)) < 1e-4

    def test_constrained_factor_ridge_enforces_expected_directions(self) -> None:
        from scipy import stats

        from valueinvestor.scorer_improver.ml_trainer import (
            _fit_constrained_factor_ridge_numpy,
        )

        rng = np.random.default_rng(41)
        X = rng.normal(size=(512, 3))
        directions = np.asarray([1.0, -1.0, -1.0])
        target = X @ np.asarray([0.7, -1.2, 0.0])

        coef, backend = _fit_constrained_factor_ridge_numpy(
            X,
            target,
            1.0,
            directions=directions,
        )

        assert backend == "constrained_numpy"
        assert coef[0] > 0.0
        assert coef[1] < 0.0
        assert coef[2] <= 0.0
        assert coef[-1] == 0.0
        assert stats.spearmanr(X @ coef[:-1], target).statistic > 0.99

    def test_mlx_constrained_factor_ridge_matches_numpy_when_available(self) -> None:
        pytest.importorskip("mlx.core")
        from valueinvestor.scorer_improver.ml_trainer import (
            _fit_constrained_factor_ridge_mlx,
            _fit_constrained_factor_ridge_numpy,
        )

        rng = np.random.default_rng(42)
        X = rng.normal(size=(512, 5))
        directions = np.asarray([1.0, -1.0, -1.0, -1.0, -1.0])
        target = X @ np.asarray([0.5, -0.8, 0.0, -0.3, 0.0])

        mlx_coef, backend = _fit_constrained_factor_ridge_mlx(
            X,
            target,
            10.0,
            directions=directions,
        )
        numpy_coef, _ = _fit_constrained_factor_ridge_numpy(
            X,
            target,
            10.0,
            directions=directions,
        )

        assert backend == "mlx-constrained"
        assert np.max(np.abs(mlx_coef - numpy_coef)) < 1e-4

    def test_mlx_adam_ridge_backend_when_available(self) -> None:
        import numpy as np

        pytest.importorskip("mlx.core")
        from valueinvestor.scorer_improver.ml_trainer import _fit_ridge

        rng = np.random.default_rng(12)
        X = rng.normal(size=(128, 8)).astype("float64")
        y = rng.normal(size=128).astype("float64")

        coef, backend = _fit_ridge(X, y, ridge_lambda=10.0, backend="mlx-adam")

        assert backend == "mlx-adam"
        assert np.isfinite(coef).all()

    def test_train_ml_ranker_uses_train_only_priors_before_gate(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import ml_trainer
        from valueinvestor.scorer_improver.promotion_gate import (
            HoldoutGateConfig,
            make_holdout_split,
        )

        rows = []
        dates = pd.date_range("2020-01-01", periods=12, freq="MS").date
        for date_idx, snap_date in enumerate(dates):
            for ticker_idx in range(20):
                rank = ticker_idx / 19.0
                rows.append(
                    {
                        "ticker": f"T{ticker_idx:03d}",
                        "snapshot_date": snap_date,
                        "close": 10.0 + ticker_idx,
                        "pe_ratio": 8.0 + rank,
                        "pe_forward": 7.5 + rank,
                        "pb_ratio": 1.0 + rank,
                        "ps_ratio": 0.5 + rank,
                        "peg_ratio": 0.8 + rank,
                        "dividend_yield": 0.01 + rank / 100.0,
                        "ev_to_ebitda": 5.0 + rank,
                        "market_cap_rmb": 10_000_000_000.0 + ticker_idx * 1_000_000.0,
                        "revenue": 100_000_000.0 + ticker_idx,
                        "net_income": 10_000_000.0 + ticker_idx,
                        "total_assets": 200_000_000.0 + ticker_idx,
                        "total_liabilities": 50_000_000.0,
                        "total_equity": 150_000_000.0,
                        "operating_cash_flow": 12_000_000.0 + ticker_idx,
                        "free_cash_flow": 8_000_000.0 + ticker_idx,
                        "gross_margin": 0.30 + rank / 10.0,
                        "roe": 0.10 + rank / 10.0,
                        "roa": 0.05 + rank / 20.0,
                        "net_margin": 0.08 + rank / 20.0,
                        "debt_to_equity": 0.30,
                        "current_ratio": 1.5,
                        "composite_score": 100.0 - ticker_idx,
                        "value_score": 50.0 + ticker_idx,
                        "quality_score": 50.0 + ticker_idx,
                        "growth_score": 50.0 + ticker_idx,
                        "forward_return_1m": rank + date_idx * 0.001,
                        "forward_return_3m": rank + date_idx * 0.002,
                        "forward_return_6m": rank + date_idx * 0.003,
                        "target_rank_1m": rank * 2.0 - 1.0,
                        "target_rank_3m": rank * 2.0 - 1.0,
                        "target_rank_6m": rank * 2.0 - 1.0,
                    }
                )
        snapshots = pd.DataFrame(rows)
        config = HoldoutGateConfig(
            holdout_months=2,
            embargo_days=20,
            min_6m_delta=-999.0,
            max_horizon_degradation=999.0,
            min_weighted_utility=-999.0,
            min_gate_snapshots=2,
            min_train_snapshots=4,
        )
        expected_split = make_holdout_split(snapshots, config=config)
        seen: dict[str, object] = {}
        original_ticker_priors = ml_trainer._ticker_priors

        def spy_ticker_priors(frame):
            seen["max_snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).max().date()
            seen["rows"] = len(frame)
            return original_ticker_priors(frame)

        monkeypatch.setattr(ml_trainer, "prepare_ml_training_snapshots", lambda **kwargs: snapshots)
        monkeypatch.setattr(ml_trainer, "_ticker_priors", spy_ticker_priors)
        monkeypatch.setattr(ml_trainer, "append_gate_ledger", lambda *args, **kwargs: None)

        metadata = ml_trainer.train_ml_ranker(
            output_model_path=tmp_path / "model.json",
            backend="numpy",
            model_kind="ridge-only",
            ridge_lambda=10.0,
            target_improvement=-999.0,
            max_training_rows=0,
            gate_config=config,
            snapshot_frequency="quarterly",
            strict_outer_gate=False,
        )

        assert seen["rows"] == len(expected_split.train)
        assert seen["max_snapshot_date"] == pd.to_datetime(expected_split.train["snapshot_date"]).max().date()
        assert "promotion_gate" in metadata
        assert metadata["promotion_gate"]["accepted"] is True

    def test_one_week_ml_ranker_skips_rolling_priors_and_records_gate_attempts(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import ml_trainer
        from valueinvestor.scorer_improver.promotion_gate import HoldoutGateConfig

        rows = []
        dates = pd.date_range("2020-01-01", periods=12, freq="MS").date
        for date_idx, snap_date in enumerate(dates):
            for ticker_idx in range(20):
                rank = ticker_idx / 19.0
                target = rank * 2.0 - 1.0
                rows.append(
                    {
                        "ticker": f"T{ticker_idx:03d}",
                        "snapshot_date": snap_date,
                        "close": 10.0 + ticker_idx,
                        "pe_ratio": 8.0 + rank,
                        "pe_forward": 7.5 + rank,
                        "pb_ratio": 1.0 + rank,
                        "ps_ratio": 0.5 + rank,
                        "peg_ratio": 0.8 + rank,
                        "dividend_yield": 0.01 + rank / 100.0,
                        "ev_to_ebitda": 5.0 + rank,
                        "market_cap_rmb": 10_000_000_000.0 + ticker_idx * 1_000_000.0,
                        "revenue": 100_000_000.0 + ticker_idx,
                        "net_income": 10_000_000.0 + ticker_idx,
                        "total_assets": 200_000_000.0 + ticker_idx,
                        "total_liabilities": 50_000_000.0,
                        "total_equity": 150_000_000.0,
                        "operating_cash_flow": 12_000_000.0 + ticker_idx,
                        "free_cash_flow": 8_000_000.0 + ticker_idx,
                        "gross_margin": 0.30 + rank / 10.0,
                        "roe": 0.10 + rank / 10.0,
                        "roa": 0.05 + rank / 20.0,
                        "net_margin": 0.08 + rank / 20.0,
                        "debt_to_equity": 0.30,
                        "current_ratio": 1.5,
                        "composite_score": 100.0 - ticker_idx,
                        "value_score": 50.0 + ticker_idx,
                        "quality_score": 50.0 + ticker_idx,
                        "growth_score": 50.0 + ticker_idx,
                        "forward_return_1w": rank + date_idx * 0.001,
                        "forward_return_1m": rank + date_idx * 0.001,
                        "forward_return_3m": rank + date_idx * 0.002,
                        "forward_return_6m": rank + date_idx * 0.003,
                        "target_rank_1w": target,
                        "target_rank_1m": target,
                        "target_rank_3m": target,
                        "target_rank_6m": target,
                    }
                )
        snapshots = pd.DataFrame(rows)

        def fail_if_rolling_priors(*args, **kwargs):
            raise AssertionError("1w training should not use rolling row priors by default")

        monkeypatch.setattr(ml_trainer, "prepare_ml_training_snapshots", lambda **kwargs: snapshots)
        monkeypatch.setattr(ml_trainer, "_attach_asof_ticker_priors", fail_if_rolling_priors)
        monkeypatch.setattr(ml_trainer, "append_gate_ledger", lambda *args, **kwargs: None)

        metadata = ml_trainer.train_ml_ranker(
            output_model_path=tmp_path / "model_1w.json",
            backend="numpy",
            ridge_lambda=10.0,
            target_improvement=-999.0,
            max_training_rows=0,
            gate_config=HoldoutGateConfig(
                holdout_months=2,
                embargo_days=0,
                min_6m_delta=-999.0,
                max_horizon_degradation=999.0,
                min_weighted_utility=-999.0,
                min_gate_snapshots=2,
                min_train_snapshots=4,
            ),
            walk_forward_folds=0,
            target_horizon="1w",
            full_eval_candidate_limit=2,
            snapshot_frequency="quarterly",
            strict_outer_gate=False,
        )

        assert metadata["target_horizon"] == "1w"
        assert metadata["target"] in {"target_rank_1w", "target_rank_1w_market"}
        assert metadata["prior_strategy"] in {"ticker_priors", "no_ticker_priors"}
        assert metadata["full_eval_candidate_limit"] == 2
        assert metadata["promotion_gate_attempts"]
        assert metadata["promotion_gate_attempts"][-1]["accepted"] is True

    def test_get_llm_client_uses_deepseek_config(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver.agent import _get_llm_client

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url=None, max_retries=0):
                self.api_key = api_key
                self.base_url = f"{base_url.rstrip('/')}/" if base_url else None
                self.max_retries = max_retries
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kwargs: None)
                )

        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_MODEL", "DeepSeek-V4-Flash")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_test_key")

        with patch("valueinvestor.analysis.llm_client.OpenAI", FakeOpenAI):
            client = _get_llm_client()

        assert client.provider == "deepseek"
        assert client.model == "deepseek-v4-flash"
        assert client._client.base_url == "https://api.deepseek.com/v1/"

    def test_run_improvement_loop_smoke_with_deepseek(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent
        from valueinvestor.scorer_improver.experiment_log import ExperimentLog

        scorer_code = textwrap.dedent("""\
        class MultiFactorScorer:
            def score(self, result):
                return result
        """).strip()
        program_md = tmp_path / "program.md"
        program_md.write_text(
            "Current:\n{current_status}\n\nHistory:\n{experiment_history}\n",
            encoding="utf-8",
        )
        exp_path = tmp_path / "experiments.jsonl"

        class TempExperimentLog(ExperimentLog):
            def __init__(self, path=None) -> None:
                super().__init__(path=exp_path)

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url=None, max_retries=0):
                self.api_key = api_key
                self.base_url = f"{base_url.rstrip('/')}/" if base_url else None
                self.max_retries = max_retries
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                return SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=f"```python\n{scorer_code}\n```\nNo changes needed."
                            )
                        )
                    ],
                )

        class FakeLLMClient:
            """Fake LLMClient with .provider attribute for the parallel path."""
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None):
                return f"```python\n{scorer_code}\n```\nNo changes needed."

        class FakeEvalContext:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def evaluate_all_targets(self, scorer_module=None, scorer_path=None):
                return {
                    "1m": {"spearman_rho": 0.12, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                    "3m": {"spearman_rho": 0.23, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                    "6m": {"spearman_rho": 0.34, "hit_rate_top20": 0.5, "mean_excess_return": 0.01},
                }

            def quick_evaluate(self, scorer_module=None, scorer_path=None, sample_size=1000):
                return {"1m": 0.11, "3m": 0.22, "6m": 0.33}

        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_MODEL", "DeepSeek-V4-Flash")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek_test_key")
        monkeypatch.setenv("IMPROVE_SCORER_PARALLEL", "1")
        monkeypatch.setattr(agent, "PROGRAM_MD_PATH", program_md)
        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path / "proposal_debug")
        monkeypatch.setattr(agent, "ExperimentLog", TempExperimentLog)
        monkeypatch.setattr(
            agent,
            "_create_parallel_clients",
            lambda: [FakeLLMClient()],
        )
        monkeypatch.setattr(agent, "ScorerEvaluationContext", FakeEvalContext)
        monkeypatch.setattr(agent, "_read_scorer", lambda: scorer_code)
        writes: list[str] = []
        monkeypatch.setattr(agent, "_write_scorer", lambda code: writes.append(code))
        monkeypatch.setattr(agent, "_validate_scorer", lambda **kwargs: True)

        with patch("valueinvestor.analysis.llm_client.OpenAI", FakeOpenAI):
            agent.run_improvement_loop(max_iterations=1)

        records = ExperimentLog(path=exp_path).read_all()
        assert len(records) == 1
        assert records[0]["kept"] is False
        assert records[0]["description"]  # should have a reason logged
        assert writes == []

    def test_try_apply_search_replace_applies_exact_match(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_search_replace

        base = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        response = textwrap.dedent("""\
            <<<<<<< SEARCH
            def bar():
                return 2
            =======
            def bar():
                return 42
            >>>>>>> REPLACE
        """)
        result = _try_apply_search_replace(base, response)
        assert result == "def foo():\n    return 1\n\ndef bar():\n    return 42\n"

    def test_try_apply_search_replace_handles_multiple_blocks(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_search_replace

        base = "a = 1\nb = 2\nc = 3\n"
        response = textwrap.dedent("""\
            <<<<<<< SEARCH
            a = 1
            =======
            a = 10
            >>>>>>> REPLACE

            Some prose here.

            <<<<<<< SEARCH
            c = 3
            =======
            c = 30
            >>>>>>> REPLACE
        """)
        result = _try_apply_search_replace(base, response)
        assert result == "a = 10\nb = 2\nc = 30\n"

    def test_try_apply_search_replace_tolerates_extra_newlines(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_search_replace

        base = "x = 1\ny = 2\nz = 3\n"
        # LLM might add leading/trailing newlines in the block
        response = "<<<<<<< SEARCH\ny = 2\n=======\ny = 20\n>>>>>>> REPLACE"
        result = _try_apply_search_replace(base, response)
        assert result == "x = 1\ny = 20\nz = 3\n"

    def test_extract_proposal_code_prefers_search_replace(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_proposal_code

        old = "x = 1\n"
        response = textwrap.dedent("""\
            <<<<<<< SEARCH
            x = 1
            =======
            x = 2
            >>>>>>> REPLACE

            ```diff
            --- a
            +++ b
            @@ -1 +1 @@
            -x = 1
            +x = 3
            ```
        """)
        # Should take the S/R block (x=2) over the diff (x=3)
        code, _, _ = _extract_proposal_code(response, old)
        assert code == "x = 2\n"

    def test_extract_code_python_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = textwrap.dedent("""\
        Here's the improved scorer:
        ```python
        class MultiFactorScorer:
            pass
        ```
        This improves the scoring by ...
        """)
        code = _extract_code(response)
        assert code is not None
        assert "class MultiFactorScorer:" in code

    def test_semantic_failure_theme_does_not_block_normal_pe_safe_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = "Removed the pe_safe scaling from the quality raw metric to avoid valuation overlap."

        assert _semantic_failure_theme(text) == ""

    def test_semantic_failure_theme_blocks_pe_safe_bug_repairs(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = "Fixed the undefined pe_safe NameError in _compute_quality_raw."

        assert _semantic_failure_theme(text) == "pe_safe_quality_raw_repair"

    def test_semantic_failure_theme_blocks_protected_value_removals(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        assert (
            _semantic_failure_theme("Removed the price × market_cap sub-factor from value scoring")
            == "remove_price_market_cap_subfactor"
        )
        assert (
            _semantic_failure_theme("Drop the price * PB interaction because it looks noisy")
            == "remove_price_pb_subfactor"
        )
        assert (
            _semantic_failure_theme("Delete the dividend-yield triangular sub-factor")
            == "remove_dividend_yield_subfactor"
        )
        assert (
            _semantic_failure_theme("Raised the ideal price * pb threshold in the log-score")
            == "widen_price_pb_threshold"
        )
        assert (
            _semantic_failure_theme("Switched the price × PB sub-factor scoring from logarithmic to linear")
            == "price_pb_shape_retry"
        )
        assert (
            _semantic_failure_theme("_log_score(ppb, best=4.0, worst=250.0)")
            == "price_pb_shape_retry"
        )
        assert (
            _semantic_failure_theme("Raised the quality ROE cap from 0.30 to 0.40")
            == "raise_quality_roe_cap"
        )
        assert (
            _semantic_failure_theme("Fix duplicate weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))")
            == "remove_earnings_yield_zero_append"
        )
        assert (
            _semantic_failure_theme("Replaced min(value_score, quality_score) synergy term with geometric mean")
            == "replace_synergy_min_retry"
        )
        assert (
            _semantic_failure_theme("Increased the influence of the absolute growth score from 0.3 to 0.4")
            == "growth_blend_weight_retry"
        )
        assert (
            _semantic_failure_theme("Increased quality_score cross-sectional percentile component from 0.8 to 0.9")
            == "quality_percentile_blend_retry"
        )
        assert _semantic_failure_theme("Reduced boost_alpha from 1.55 to 1.45") == "value_boost_alpha_retry"
        assert (
            _semantic_failure_theme("Added a gross-profit-yield (gross profit / market cap) value factor")
            == "gross_profit_market_cap_yield_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the market-cap gate for the gross-profit-to-assets value sub-factor")
            == "gross_profit_assets_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the linear scoring of asset turnover in _momentum_score")
            == "asset_turnover_retry"
        )
        assert (
            _semantic_failure_theme("Raised the best threshold of the operating-cash-flow yield")
            == "ocf_yield_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the log‑scale scoring of operating‑cash‑flow yield")
            == "ocf_yield_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Shifted the optimal EV/EBITDA from 8.0 to 9.0")
            == "ev_ebitda_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Widened the PEG scoring reference window")
            == "peg_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Increased the loss penalty scaling constant from 5.0 to 6.0")
            == "loss_penalty_scaling_retry"
        )
        assert (
            _semantic_failure_theme("The duplicate zero-score entry now uses a smaller weight")
            == "earnings_yield_zero_weight_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap composite score bonus gate")
            == "large_cap_composite_bonus_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the best threshold for the ROA value sub-factor from 0.15 to 0.12")
            == "roa_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap post-percentile quality_score ROA bonus")
            == "large_cap_quality_bonus_retry"
        )
        assert (
            _semantic_failure_theme("Added a large-cap-gated direct ROE value sub-factor")
            == "large_cap_direct_roe_value_retry"
        )
        assert (
            _semantic_failure_theme(
                "Lowered the market-cap gate for the liabilities-to-market-cap value sub-factor"
            )
            == "liabilities_market_cap_gate_retry"
        )
        assert (
            _semantic_failure_theme("Replaced the linear scoring of roe_ey with a logarithmic shape")
            == "roe_ev_ebitda_value_shape_retry"
        )
        assert (
            _semantic_failure_theme("Added a fallback to pe_forward when PEG is unavailable")
            == "growth_forward_pe_fallback_retry"
        )
        assert (
            _semantic_failure_theme("Lowered the best threshold for the forward P/E log-score")
            == "forward_pe_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Raised the forward_pe_improve linear_score best threshold")
            == "forward_pe_value_threshold_retry"
        )
        assert (
            _semantic_failure_theme("Added a value_score > 60 and quality_score > 60 composite_score bonus")
            == "value_quality_composite_bonus_retry"
        )

    def test_semantic_failure_theme_ignores_unchanged_diff_context(self) -> None:
        from valueinvestor.scorer_improver.agent import _semantic_failure_theme

        text = """\
--- old
+++ new
@@ -442,2 +442,12 @@
                     weighted_scores.append((0.0, CURRENT_RATIO_WEIGHT))
+        # ROE / PB - profitable book value
+        roe_pb = roe_val / pb_val
Added a new value sub-factor ROE / PB that rewards profitability relative to book value.
"""

        assert _semantic_failure_theme(text) == ""

    def test_best_scorer_snapshot_round_trip(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("BEST = True\n", encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        agent._save_best_scorer_snapshot(
            iteration=42,
            rhos={"1m": 0.11, "3m": 0.12, "6m": 0.13},
            ground_truth_id="gt-current",
        )
        scorer_path.write_text("BEST = False\n", encoding="utf-8")

        meta = agent._restore_best_scorer_snapshot("gt-current")

        assert scorer_path.read_text(encoding="utf-8") == "BEST = True\n"
        assert meta is not None
        assert meta["iteration"] == 42
        assert json.loads(meta_path.read_text(encoding="utf-8"))["spearman_rho_6m"] == 0.13

    def test_cleanup_old_backups_preserves_tracked_files(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        tracked = backup_dir / "scorer_0000.py"
        old_untracked = backup_dir / "scorer_0001.py"
        new_untracked = backup_dir / "scorer_0002.py"
        for path in (tracked, old_untracked, new_untracked):
            path.write_text(path.name, encoding="utf-8")

        monkeypatch.setattr(agent, "BACKUP_DIR", backup_dir)
        monkeypatch.setattr(agent, "_MAX_BACKUPS", 1)
        monkeypatch.setattr(agent, "_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            agent.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout="backups/scorer_0000.py\n",
            ),
        )

        assert agent._cleanup_old_backups() == 1
        assert tracked.exists()
        assert not old_untracked.exists()
        assert new_untracked.exists()

    def test_best_scorer_snapshot_ignores_other_ground_truth(self, tmp_path: Path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("CURRENT = True\n", encoding="utf-8")
        best_path.write_text("BEST = True\n", encoding="utf-8")
        meta_path.write_text(json.dumps({"ground_truth_id": "old-gt"}), encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        assert agent._restore_best_scorer_snapshot("new-gt") is None
        assert scorer_path.read_text(encoding="utf-8") == "CURRENT = True\n"

    def test_load_resume_baseline_uses_snapshot_without_mutating_worktree(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from valueinvestor.scorer_improver import agent

        scorer_path = tmp_path / "scorer.py"
        best_path = tmp_path / "current_best_scorer.py"
        meta_path = tmp_path / "current_best_scorer.json"
        scorer_path.write_text("WORKTREE = True\n", encoding="utf-8")
        best_path.write_text("BEST = True\n", encoding="utf-8")
        meta_path.write_text(json.dumps({"ground_truth_id": "gt-current"}), encoding="utf-8")

        monkeypatch.setattr(agent, "SCORER_PATH", scorer_path)
        monkeypatch.setattr(agent, "BEST_SCORER_PATH", best_path)
        monkeypatch.setattr(agent, "BEST_SCORER_META_PATH", meta_path)

        meta, code, from_snapshot = agent._load_resume_baseline("gt-current")

        assert from_snapshot is True
        assert meta == {"ground_truth_id": "gt-current"}
        assert code == "BEST = True\n"
        assert scorer_path.read_text(encoding="utf-8") == "WORKTREE = True\n"

    def test_historical_snapshot_guard_only_saves_true_bests(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_better_historical_snapshot

        best = {"1m": 0.11, "3m": 0.12, "6m": 0.13}

        assert _is_better_historical_snapshot(
            {"1m": 0.1102, "3m": 0.119, "6m": 0.129},
            best,
        )
        assert not _is_better_historical_snapshot(
            {"1m": 0.11001, "3m": 0.119, "6m": 0.129},
            best,
        )
        assert not _is_better_historical_snapshot(
            {"1m": 0.10, "3m": 0.119, "6m": 0.129},
            best,
        )

    def test_print_skip_banner_shows_each_horizon(self, capsys) -> None:
        from valueinvestor.scorer_improver.agent import _print_skip_banner

        rhos = {"1m": 0.0455, "3m": 0.0878, "6m": 0.1035}
        best = {"1m": 0.1646, "3m": 0.1771, "6m": 0.2430}

        _print_skip_banner(7992, rhos, best, "no improvement")

        output = capsys.readouterr().out
        assert "Iteration #7992  |  ❌ REVERTED (no improvement)" in output
        assert "1-month  ρ=0.0455  |  Best: 0.1646" in output
        assert "3-month  ρ=0.0878  |  Best: 0.1771" in output
        assert "6-month  ρ=0.1035  |  Best: 0.2430" in output

    def test_extract_code_no_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = "No code block here, just plain text."
        code = _extract_code(response)
        assert code is None

    def test_extract_code_generic_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_code

        response = textwrap.dedent("""\
        ```
        print("hello")
        ```
        """)
        code = _extract_code(response)
        assert code is not None
        assert "print" in code

    def test_extract_patch_diff_fence(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_patch

        response = textwrap.dedent("""\
        ```diff
        --- a
        +++ b
        @@ -1 +1 @@
        -old
        +new
        ```
        Changed one line.
        """)
        patch = _extract_patch(response)
        assert patch is not None
        assert "@@ -1 +1 @@" in patch
        assert "+new" in patch

    def test_extract_proposal_code_prefers_patch(self) -> None:
        from valueinvestor.scorer_improver.agent import (
            _compute_full_diff,
            _extract_proposal_code,
        )

        old = "def score():\n    return 1\n"
        new = "def score():\n    return 2\n"
        patch = _compute_full_diff(old, new)
        response = f"```diff\n{patch}```\nUse a stronger score."

        code, explanation, diff_summary = _extract_proposal_code(response, old)

        assert code == new
        assert "stronger" in explanation.lower()
        assert "return 2" in diff_summary

    def test_extract_proposal_code_handles_indented_diff(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_proposal_code

        old = "def score():\n    return 1\n"
        response = textwrap.dedent("""\
            ```diff
                --- a
                +++ b
                @@ -1,2 +1,2 @@
                 def score():
                -    return 1
                +    return 4
            ```
            Repaired patch.
        """)

        code, _, diff_summary = _extract_proposal_code(response, old)

        assert code == "def score():\n    return 4\n"
        assert "return 4" in diff_summary

    def test_extract_proposal_code_rejects_full_file_fallback(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_proposal_code

        old = "def score():\n    return 1\n"
        new = "def score():\n    return 3\n"
        response = f"```python\n{new}```\nFallback full file."

        code, explanation, diff_summary = _extract_proposal_code(response, old)

        assert code is None
        assert "fallback" in explanation.lower()
        assert diff_summary == ""

    def test_extract_explanation(self) -> None:
        from valueinvestor.scorer_improver.agent import _extract_explanation

        response = textwrap.dedent("""\
        ```python
        class Foo:
            pass
        ```
        I changed the weights to be more balanced.
        """)
        explanation = _extract_explanation(response)
        assert "weights" in explanation.lower() or "balanced" in explanation.lower()

    def test_compute_diff_summary(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_diff_summary

        old = "line1\nline2\nline3"
        new = "line1\nmodified\nline3"
        diff = _compute_diff_summary(old, new)
        assert "line2" in diff or "modified" in diff
        assert len(diff) <= 500

    def test_compute_diff_no_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_diff_summary

        code = "same code"
        diff = _compute_diff_summary(code, code)
        assert "no changes" in diff.lower()

    def test_compute_full_diff(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff

        old = "line1\nline2\nline3\n"
        new = "line1\nmodified\nline3\n"
        diff = _compute_full_diff(old, new)
        assert "line2" in diff or "modified" in diff

    def test_try_apply_patch_applies_clean_change(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        old = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        new = "def foo():\n    x = 1\n    y = 3\n    return x + y\n"
        patch = _compute_full_diff(old, new)
        result = _try_apply_patch(old, patch)
        assert result == new

    def test_try_apply_patch_stacks_independent_changes(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        base = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        change1 = "def a():\n    return 10\n\ndef b():\n    return 2\n"
        change2 = "def a():\n    return 1\n\ndef b():\n    return 20\n"

        # Apply change1
        patch1 = _compute_full_diff(base, change1)
        after1 = _try_apply_patch(base, patch1)
        assert after1 == change1

        # Stack change2 on top
        patch2 = _compute_full_diff(base, change2)  # diff vs original base
        stacked = _try_apply_patch(after1, patch2)
        expected = "def a():\n    return 10\n\ndef b():\n    return 20\n"
        assert stacked == expected

    def test_try_apply_patch_tolerates_stale_line_numbers(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        base = (
            "def first():\n"
            "    return 1\n"
            "\n"
            "def target():\n"
            "    score = 2\n"
            "    return score\n"
        )
        patch = textwrap.dedent("""\
            --- a/src/valueinvestor/screener/scorer.py
            +++ b/src/valueinvestor/screener/scorer.py
            @@ -1,3 +1,3 @@
             def target():
            -    score = 2
            +    score = 5
        """)

        result = _try_apply_patch(base, patch)

        assert result is not None
        assert "score = 5" in result
        assert "def first" in result

    def test_try_apply_patch_tolerates_omitted_blank_context(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        base = (
            "def quality():\n"
            "    pe = result.valuation.pe_ratio\n"
            "\n"
            "    # Require all inputs\n"
            "    if pe is None:\n"
            "        return None\n"
        )
        patch = textwrap.dedent("""\
            --- a/scorer.py
            +++ b/scorer.py
            @@ -1,5 +1,6 @@
             def quality():
                 pe = result.valuation.pe_ratio
            +    current_ratio = result.financials.current_ratio
                 # Require all inputs
                 if pe is None:
                     return None
        """)

        result = _try_apply_patch(base, patch)

        assert result is not None
        assert "current_ratio" in result

    def test_try_apply_patch_returns_none_on_context_mismatch(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_full_diff, _try_apply_patch

        old = "def foo():\n    return 1\n"
        new = "def foo():\n    return 2\n"
        patch = _compute_full_diff(old, new)
        different_base = "def bar():\n    return 3\n"
        assert _try_apply_patch(different_base, patch) is None

    def test_try_apply_patch_returns_none_without_hunks(self) -> None:
        from valueinvestor.scorer_improver.agent import _try_apply_patch

        assert _try_apply_patch("x = 1\n", "This is not a patch") is None

    def test_records_for_prompt_skips_no_patch_noise(self) -> None:
        from valueinvestor.scorer_improver.agent import _records_for_prompt

        records = [
            {"iteration": 1, "kept": False, "description": "prose only", "diff_summary": ""},
            {"iteration": 2, "kept": False, "description": "actual patch", "diff_summary": "--- old\n+++ new\n"},
        ]

        assert [r["iteration"] for r in _records_for_prompt(records, 10)] == [2]

    def test_failure_guidance_summarizes_quick_reject_patterns(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_failure_guidance

        records = [
            {
                "kept": False,
                "quick_reject_reason": "1m-only tradeoff",
                "quick_material_improved_horizons": ["1m"],
                "quick_material_degraded_horizons": ["3m", "6m"],
            },
            {
                "kept": False,
                "quick_reject_reason": "1m-only tradeoff",
                "quick_material_improved_horizons": ["1m"],
                "quick_material_degraded_horizons": ["6m"],
            },
        ]

        guidance = _compute_failure_guidance(records)

        assert "1m-only tradeoff" in guidance
        assert "1m upside paired with 3m/6m damage" in guidance

    def test_near_miss_guidance_targets_1m_repair(self) -> None:
        from valueinvestor.scorer_improver.agent import _near_miss_guidance

        records = [
            {
                "iteration": 13743,
                "kept": False,
                "near_miss": True,
                "current_material_improved_horizons": ["3m", "6m"],
                "current_material_degraded_horizons": ["1m"],
                "current_delta_1m": -0.001176,
                "current_delta_3m": 0.000453,
                "current_delta_6m": 0.000106,
                "diff_summary": "PB best threshold 20.0 -> 22.0",
            },
        ]

        guidance = _near_miss_guidance(records)

        assert "Near-miss repair objective" in guidance
        assert "repairing 1m damage" in guidance
        assert "#13743" in guidance
        assert "Δ3m=+0.0005" in guidance

    def test_failure_guidance_promotes_blocked_themes_to_hard_exclusions(self) -> None:
        from valueinvestor.scorer_improver.agent import _blocked_failure_themes, _compute_failure_guidance

        records = [
            {
                "kept": False,
                "description": "Added FCF/TA value sub-factor",
                "diff_summary": "fcf / total_assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Added free cash flow to total assets yield",
                "diff_summary": "free cash flow to total assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Reward FCF/equity in value score",
                "diff_summary": "fcf_equity_yield",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        assert _blocked_failure_themes(records) == ["fcf_assets_or_equity_retry"]
        guidance = _compute_failure_guidance(records)
        assert "Hard proposal exclusions" in guidance
        assert "FCF/assets, FCF/equity, or FCF/debt retry variants" in guidance

    def test_failure_guidance_uses_wider_blocked_theme_window(self) -> None:
        from valueinvestor.scorer_improver.agent import _compute_failure_guidance

        blocked_records = [
            {
                "kept": False,
                "description": "Lowered the best threshold for forward P/E",
                "diff_summary": "pe_forward best threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Raised the forward_pe_improve linear_score best threshold",
                "diff_summary": "forward_pe_improve threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Adjusted forward-pe improvement threshold",
                "diff_summary": "forward-pe improvement threshold",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        guidance = _compute_failure_guidance([], blocked_theme_records=blocked_records)

        assert "Hard proposal exclusions" in guidance
        assert "forward-PE value threshold" in guidance

    def test_llm_complete_passes_request_timeout_when_supported(self) -> None:
        from valueinvestor.scorer_improver.agent import _llm_complete

        class FakeLLM:
            def __init__(self) -> None:
                self.timeout = None

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None, timeout=None):
                self.timeout = timeout
                return "ok"

        fake = FakeLLM()

        assert _llm_complete(fake, "prompt", 10, request_timeout=2.5) == "ok"
        assert fake.timeout == 2.5

    def test_llm_complete_falls_back_for_test_doubles_without_timeout(self) -> None:
        from valueinvestor.scorer_improver.agent import _llm_complete

        class FakeLLM:
            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return "ok"

        assert _llm_complete(FakeLLM(), "prompt", 10, request_timeout=2.5) == "ok"

    def test_llm_complete_empty_retry_stays_bounded(self) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None, timeout=None):
                self.calls.append(max_tokens)
                return "" if len(self.calls) == 1 else "ok"

        fake = FakeLLM()

        assert agent._llm_complete(fake, "prompt", 6000) == "ok"
        assert fake.calls == [6000, agent._MAX_REPAIR_TOKENS]

    def test_repeated_failed_theme_detects_pe_safe_repairs(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Fixed undefined pe_safe in _compute_quality_raw",
                "diff_summary": "pe_safe",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Repair broken quality raw bug",
                "diff_summary": "_compute_quality_raw",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Remove fatal bug in quality raw",
                "diff_summary": "undefined",
                "delta_1m": 0.0,
                "delta_3m": 0.0,
                "delta": 0.0,
            },
        ]

        theme = _is_repeated_failed_theme(
            "replace pe_safe with pe ratio",
            "fix undefined _compute_quality_raw",
            recent,
        )

        assert theme == "pe_safe_quality_raw_repair"

    def test_repeated_failed_theme_detects_protected_value_removals(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Removed price × market_cap from value score",
                "diff_summary": "- PRICE_MARKET_CAP_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Drop price * market_cap because it overlaps",
                "diff_summary": "- pmc = price * market_cap",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Remove price_market_cap_weight sub-factor",
                "diff_summary": "PRICE_MARKET_CAP_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "- PRICE_MARKET_CAP_WEIGHT",
            "Removed the price x market_cap sub-factor",
            recent,
        )

        assert theme == "remove_price_market_cap_subfactor"

    def test_repeated_failed_theme_detects_earnings_yield_zero_append_removal(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Removed duplicate earnings yield zero append",
                "diff_summary": "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
                "delta_1m": -0.02,
                "delta_3m": -0.01,
                "delta": -0.03,
            },
            {
                "kept": False,
                "description": "Delete redundant earnings_yield_weight zero branch",
                "diff_summary": "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Fix double earnings yield zero append",
                "diff_summary": "EARNINGS_YIELD_WEIGHT duplicate",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "- weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))",
            "Remove duplicate earnings yield zero append",
            recent,
        )

        assert theme == "remove_earnings_yield_zero_append"

    def test_repeated_failed_theme_detects_fcf_assets_retries(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Added FCF/TA value sub-factor",
                "diff_summary": "FCF_TO_ASSETS_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Added free cash flow to total assets yield",
                "diff_summary": "fcf / total_assets",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Reward FCF/equity in value score",
                "diff_summary": "fcf_equity_yield",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "fcf_ta = fcf / ta",
            "Add a free cash flow to total assets value factor",
            recent,
        )

        assert theme == "fcf_assets_or_equity_retry"

    def test_repeated_failed_theme_detects_current_ratio_retries(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_repeated_failed_theme

        recent = [
            {
                "kept": False,
                "description": "Added current-ratio triangular bonus",
                "diff_summary": "current_ratio",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Large-cap gated current ratio factor",
                "diff_summary": "CURRENT_RATIO_WEIGHT",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
            {
                "kept": False,
                "description": "Quality raw current ratio gate",
                "diff_summary": "cr = result.financials.current_ratio",
                "delta_1m": -0.01,
                "delta_3m": -0.01,
                "delta": -0.01,
            },
        ]

        theme = _is_repeated_failed_theme(
            "current_ratio = result.financials.current_ratio",
            "Add a current ratio bonus",
            recent,
        )

        assert theme == "current_ratio_retry"

    def test_behavior_neutral_detects_tiny_full_eval_deltas(self) -> None:
        from valueinvestor.scorer_improver.agent import _is_behavior_neutral

        baseline = {"1m": 0.10, "3m": 0.20, "6m": 0.30}

        assert _is_behavior_neutral({"1m": 0.10002, "3m": 0.19998, "6m": 0.30001}, baseline)
        assert not _is_behavior_neutral({"1m": 0.1002, "3m": 0.20, "6m": 0.30}, baseline)

    def test_agent_prompt_describes_current_pareto_acceptance_gate(self) -> None:
        from valueinvestor.scorer_improver.agent import _AGENT_SYSTEM_PROMPT, _build_prompt

        assert "three time horizons" in _AGENT_SYSTEM_PROMPT
        assert "SEARCH/REPLACE" in _AGENT_SYSTEM_PROMPT

        prompt = _build_prompt(
            scorer_code="class MultiFactorScorer:\n    pass\n",
            program_md="Status:\n{current_status}\n\nHistory:\n{experiment_history}\n",
            experiment_history="No history",
            baseline_rhos={"1m": 0.10, "3m": 0.20, "6m": 0.30},
            best_rhos={"1m": 0.10, "3m": 0.20, "6m": 0.30},
        )

        assert "Acceptance gate: improve at least one horizon by more than 0.0001" in prompt
        assert "SEARCH/REPLACE block" in prompt
        assert "```python" in prompt

    def test_default_weight_guard_accepts_full_six_factor_weights(self) -> None:
        from valueinvestor.scorer_improver.agent import _default_weight_issues

        code = """
_DEFAULT_WEIGHTS = {
    "value": 0.40,
    "quality": 0.20,
    "growth": 0.10,
    "momentum": 0.10,
    "synergy": 0.10,
    "value_growth": 0.10,
}
"""

        assert _default_weight_issues(code) == []

    def test_default_weight_guard_rejects_partial_weights(self) -> None:
        from valueinvestor.scorer_improver.agent import _default_weight_issues

        code = """
_DEFAULT_WEIGHTS = {
    "value": 0.70,
    "quality": 0.20,
    "growth": 0.10,
}
"""

        issues = _default_weight_issues(code)

        assert any("momentum" in issue for issue in issues)
        assert any("synergy" in issue for issue in issues)

    def test_quick_reject_diagnostics_rejects_1m_only_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1370, "3m": 0.1370, "6m": 0.1660}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "1m-only tradeoff"
        assert diagnostics["material_improved_horizons"] == ["1m"]
        assert diagnostics["material_degraded_horizons"] == ["3m", "6m"]

    def test_quick_reject_diagnostics_keeps_multi_horizon_upside(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1352, "3m": 0.1418, "6m": 0.1701}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is False
        assert diagnostics["material_improved_horizons"] == ["3m", "6m"]

    def test_quick_reject_diagnostics_rejects_severe_1m_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1368, "3m": 0.1436, "6m": 0.1724}
        candidate = {"1m": 0.1345, "3m": 0.1458, "6m": 0.1721}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "1m tradeoff"
        assert diagnostics["material_improved_horizons"] == ["3m"]
        assert "1m" in diagnostics["severe_degraded_horizons"]

    def test_quick_reject_diagnostics_tolerates_sample_noise(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1351, "3m": 0.1407, "6m": 0.1688}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is False
        assert diagnostics["material_improved_horizons"] == []
        assert diagnostics["material_degraded_horizons"] == []

    def test_quick_reject_diagnostics_rejects_no_sample_upside(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1368, "3m": 0.1436, "6m": 0.1724}
        candidate = {"1m": 0.1364, "3m": 0.1434, "6m": 0.1723}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "no sample upside"
        assert diagnostics["no_sample_upside"] is True
        assert diagnostics["material_improved_horizons"] == []

    def test_quick_reject_diagnostics_rejects_sample_noop(self) -> None:
        from valueinvestor.scorer_improver.agent import _quick_reject_diagnostics

        quick_baseline = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}
        candidate = {"1m": 0.1350, "3m": 0.1410, "6m": 0.1690}

        diagnostics = _quick_reject_diagnostics(candidate, quick_baseline)

        assert diagnostics["reject"] is True
        assert diagnostics["reason"] == "behavior-neutral sample"
        assert diagnostics["noop_sample"] is True

    def test_acceptance_diagnostics_explains_target_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.1315, "3m": 0.1355, "6m": 0.1663}
        target = {"1m": 0.1347, "3m": 0.1355, "6m": 0.1663}
        candidate = {"1m": 0.1307, "3m": 0.1360, "6m": 0.1665}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == []
        assert diagnostics["current_improved_horizons"] == ["3m", "6m"]
        assert diagnostics["target_improved_horizons"] == ["3m", "6m"]
        assert diagnostics["reject_reason"] == "target utility tradeoff"
        assert diagnostics["near_miss"] is True

    def test_acceptance_diagnostics_accepts_current_pareto_despite_historical_target(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.130593, "3m": 0.139496, "6m": 0.170561}
        target = {"1m": 0.134683, "3m": 0.139496, "6m": 0.170561}
        candidate = {"1m": 0.131434, "3m": 0.139771, "6m": 0.170960}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == ["1m", "3m", "6m"]
        assert diagnostics["accepted_by"] == "current_material_pareto"
        assert diagnostics["historical_reject_reason"] == "target utility tradeoff"
        assert diagnostics["reject_reason"] == ""
        assert diagnostics["near_miss"] is False

    def test_acceptance_diagnostics_accepts_material_improvement_with_neutral_dip(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.130593, "3m": 0.139496, "6m": 0.170561}
        target = {"1m": 0.134683, "3m": 0.139496, "6m": 0.170561}
        candidate = {"1m": 0.130790, "3m": 0.139946, "6m": 0.170533}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == ["1m", "3m"]
        assert diagnostics["accepted_by"] == "current_material_pareto"
        assert diagnostics["current_degraded_horizons"] == ["6m"]
        assert diagnostics["current_material_degraded_horizons"] == []
        assert diagnostics["historical_reject_reason"] == "target utility tradeoff"
        assert diagnostics["reject_reason"] == ""
        assert diagnostics["near_miss"] is False

    def test_acceptance_diagnostics_rejects_tiny_noise_gain(self) -> None:
        from valueinvestor.scorer_improver.agent import _acceptance_diagnostics

        current = {"1m": 0.133937, "3m": 0.143385, "6m": 0.174097}
        target = dict(current)
        candidate = {"1m": 0.133918, "3m": 0.143421, "6m": 0.174177}

        diagnostics = _acceptance_diagnostics(candidate, current, target)

        assert diagnostics["accepted_horizons"] == []
        assert diagnostics["current_material_improved_horizons"] == []
        assert diagnostics["near_miss"] is True

    def test_complete_proposal_saves_invalid_debug_artifact(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return "```diff\n--- a\n+++ b\n@@ -1 +1 @@\n-missing\n+new\n```\n"

        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", False)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(FakeLLM(), "prompt", "old\n", iteration=123)

        assert result["code"] is None
        assert result["debug_paths"]
        assert "Raw Response" in Path(result["debug_paths"][0]).read_text()

    def test_complete_proposal_rejects_syntax_invalid_code(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                return textwrap.dedent("""\
                    ```diff
                    --- a/scorer.py
                    +++ b/scorer.py
                    @@ -1,2 +1,2 @@
                    -def score():
                    -    return 1
                    +def score():
                    +return 2
                    ```
                """)

        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", False)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(FakeLLM(), "prompt", "def score():\n    return 1\n", iteration=456)

        assert result["code"] is None
        assert "Syntax-invalid proposal" in result["explanation"]
        assert result["debug_paths"]

    def test_complete_proposal_does_not_uncap_empty_repair(self, tmp_path, monkeypatch) -> None:
        from valueinvestor.scorer_improver import agent

        class FakeLLM:
            provider = "deepseek"

            def __init__(self) -> None:
                self.calls = []

            def complete(self, system_prompt=None, user_prompt=None, max_tokens=None):
                self.calls.append(max_tokens)
                if len(self.calls) == 1:
                    return "This is not a patch."
                return ""

        fake = FakeLLM()
        monkeypatch.setattr(agent, "PROPOSAL_DEBUG_DIR", tmp_path)
        monkeypatch.setattr(agent, "_REPAIR_INVALID_PROPOSALS", True)
        monkeypatch.setattr(agent, "_SAVE_INVALID_PROPOSALS", True)

        result = agent._complete_proposal(fake, "prompt", "old\n", iteration=789)

        assert result["code"] is None
        assert len(fake.calls) == 2
        assert fake.calls[0] == agent._MAX_PROPOSAL_TOKENS
        assert fake.calls[1] == agent._MAX_REPAIR_TOKENS
        assert result["debug_paths"]

    def test_accept_policy_rejects_bad_tradeoff(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.102, "3m": 0.091, "6m": 0.091}
        assert _accepted_horizons(candidate, baseline) == []

    def test_accept_policy_accepts_weighted_improvement(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.100, "3m": 0.101, "6m": 0.103}
        assert _accepted_horizons(candidate, baseline) == ["3m", "6m"]

    def test_accept_policy_tolerates_behavior_neutral_dip(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.1002, "3m": 0.09997, "6m": 0.10}
        assert _accepted_horizons(candidate, baseline) == ["1m"]

    def test_accept_policy_rejects_only_behavior_neutral_movement(self) -> None:
        from valueinvestor.scorer_improver.agent import _accepted_horizons

        baseline = {"1m": 0.10, "3m": 0.10, "6m": 0.10}
        candidate = {"1m": 0.10002, "3m": 0.10001, "6m": 0.09999}
        assert _accepted_horizons(candidate, baseline) == []


# ── Evaluator tests (with mocked ground truth) ──────────────────────────────


class TestEvaluator:
    """Tests for evaluate_scorer with synthetic ground truth."""

    def _make_gt(self) -> pd.DataFrame:
        """Create a small synthetic ground truth DataFrame."""
        import numpy as np

        np.random.seed(42)
        n = 50
        tickers = [f"SH{i:06d}" for i in range(n)]
        return pd.DataFrame({
            "ticker": tickers,
            "snapshot_date": pd.Timestamp("2023-06-01"),
            "close": np.random.uniform(10, 100, n),
            "pe_ratio": np.random.uniform(5, 50, n),
            "pb_ratio": np.random.uniform(0.5, 8, n),
            "market_cap_rmb": np.random.uniform(1e9, 1e11, n),
            "pe_forward": np.random.uniform(4, 40, n),
            "ps_ratio": np.random.uniform(0.2, 8, n),
            "peg_ratio": np.random.uniform(0.2, 3, n),
            "dividend_yield": np.random.uniform(0, 0.08, n),
            "ev_to_ebitda": np.random.uniform(2, 30, n),
            "revenue": np.random.uniform(1e8, 1e11, n),
            "net_income": np.random.uniform(-1e9, 1e10, n),
            "total_assets": np.random.uniform(1e9, 2e11, n),
            "total_liabilities": np.random.uniform(1e8, 1e11, n),
            "total_equity": np.random.uniform(1e8, 1e11, n),
            "operating_cash_flow": np.random.uniform(-1e9, 1e10, n),
            "free_cash_flow": np.random.uniform(-1e9, 1e10, n),
            "gross_margin": np.random.uniform(0, 0.8, n),
            "roa": np.random.uniform(-0.1, 0.2, n),
            "current_ratio": np.random.uniform(0.2, 4, n),
            "roe": np.random.uniform(-0.1, 0.3, n),
            "net_margin": np.random.uniform(-0.05, 0.3, n),
            "debt_to_equity": np.random.uniform(0, 3, n),
            # Make composite_score correlated with forward_return for a non-zero ρ
            "composite_score": np.random.uniform(20, 90, n),
            "value_score": np.random.uniform(20, 90, n),
            "quality_score": np.random.uniform(20, 90, n),
            "growth_score": np.random.uniform(20, 90, n),
            "forward_return_1m": np.random.uniform(-0.1, 0.2, n),
            "forward_return_3m": np.random.uniform(-0.2, 0.35, n),
            "forward_return_6m": np.random.uniform(-0.3, 0.5, n),
        })

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_evaluate_with_original_scores(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver.evaluator import evaluate_scorer

        gt = self._make_gt()
        mock_load.return_value = gt

        result = evaluate_scorer(use_original_scores=True)
        assert "spearman_rho" in result
        assert "hit_rate_top20" in result
        assert result["n_snapshots"] == 1
        # Rho should be some float (sign depends on random seed)
        assert isinstance(result["spearman_rho"], float)

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_evaluate_empty_gt(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver.evaluator import evaluate_scorer

        mock_load.return_value = pd.DataFrame()
        result = evaluate_scorer(use_original_scores=True)
        assert result["spearman_rho"] == 0.0
        assert result["n_snapshots"] == 0

    @patch("valueinvestor.scorer_improver.evaluator._load_ground_truth")
    def test_rescore_passes_full_feature_set(self, mock_load: MagicMock) -> None:
        from valueinvestor.scorer_improver import evaluator

        gt = self._make_gt().head(12).copy()
        mock_load.return_value = gt
        seen = {}

        class RecordingScorer:
            def score(self, result):
                seen["financials"] = result.financials
                seen["valuation"] = result.valuation
                result.composite_score = 42.0
                return result

            def rank(self, results):
                seen["rank_called"] = True
                for result in results:
                    self.score(result)
                return results

        with patch.object(evaluator.importlib, "reload", side_effect=lambda mod: mod):
            with patch("valueinvestor.screener.scorer.MultiFactorScorer", return_value=RecordingScorer()):
                evaluator.evaluate_scorer(use_original_scores=False)

        assert seen["valuation"].ps_ratio is not None
        assert seen["rank_called"] is True
        assert seen["valuation"].peg_ratio is not None
        assert seen["valuation"].dividend_yield is not None
        assert seen["financials"].revenue is not None
        assert seen["financials"].gross_margin is not None
        assert seen["financials"].operating_cash_flow is not None

    def test_sample_by_snapshot_keeps_cross_section_shape(self) -> None:
        from valueinvestor.scorer_improver.evaluator import _sample_by_snapshot

        frames = []
        for idx, snapshot in enumerate(pd.date_range("2023-01-01", periods=3, freq="90D")):
            frames.append(pd.DataFrame({
                "ticker": [f"TEST{idx:02d}{i:03d}" for i in range(30)],
                "snapshot_date": snapshot,
                "close": range(30),
            }))
        gt = pd.concat(frames, ignore_index=True)
        sampled = _sample_by_snapshot(gt, sample_size=30)
        counts = sampled.groupby("snapshot_date").size()
        assert len(sampled) == 30
        assert set(counts) == {10}
        assert len(counts) == 3

    def test_evaluation_context_reuses_loaded_frame_and_quick_sample(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver import evaluator
        from valueinvestor.scorer_improver.evaluator import ScorerEvaluationContext

        gt = self._make_gt()
        load_calls = []
        sample_calls = []

        def fake_load(path=None, *, allow_legacy_schema=True):
            load_calls.append((path, allow_legacy_schema))
            return gt

        def fake_sample(frame, sample_size):
            sample_calls.append(sample_size)
            return frame.head(12)

        monkeypatch.setattr(evaluator, "_load_ground_truth", fake_load)
        monkeypatch.setattr(evaluator, "_sample_by_snapshot", fake_sample)
        monkeypatch.setattr(
            evaluator,
            "_evaluate_scorer_all_targets_frame",
            lambda frame, *, scorer_module, scorer_path=None, snapshot_templates=None: {
                "1m": {"spearman_rho": 0.1},
                "3m": {"spearman_rho": 0.2},
                "6m": {"spearman_rho": 0.3},
            },
        )
        monkeypatch.setattr(
            evaluator,
            "_quick_evaluate_frame",
            lambda frame, *, scorer_module, scorer_path=None, snapshot_templates=None: {
                "1m": 0.1,
                "3m": 0.2,
                "6m": 0.3,
            },
        )

        context = ScorerEvaluationContext(quick_sample_size=12)
        context.evaluate_all_targets()
        context.evaluate_all_targets()
        context.quick_evaluate(sample_size=12)
        context.quick_evaluate(sample_size=12)

        assert len(load_calls) == 1
        assert sample_calls == [12]

    def test_evaluation_context_builds_snapshot_templates_once_per_cached_frame(self, monkeypatch) -> None:
        from valueinvestor.scorer_improver import evaluator
        from valueinvestor.scorer_improver.evaluator import ScorerEvaluationContext

        gt = pd.DataFrame({
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "snapshot_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "close": [10.0, 11.0, 12.0, 13.0],
        })
        template_calls = []

        monkeypatch.setattr(
            evaluator,
            "_load_ground_truth",
            lambda path=None, *, allow_legacy_schema=True: gt,
        )
        monkeypatch.setattr(
            evaluator,
            "_sample_by_snapshot",
            lambda frame, sample_size: frame.head(2),
        )

        def fake_build(frame):
            template_calls.append(tuple(frame.index))
            return []

        monkeypatch.setattr(evaluator, "_build_snapshot_templates", fake_build)
        monkeypatch.setattr(
            evaluator,
            "_evaluate_scorer_all_targets_frame",
            lambda frame, **kwargs: {
                "1m": {"spearman_rho": 0.1},
                "3m": {"spearman_rho": 0.2},
                "6m": {"spearman_rho": 0.3},
            },
        )
        monkeypatch.setattr(
            evaluator,
            "_quick_evaluate_frame",
            lambda frame, **kwargs: {"1m": 0.1, "3m": 0.2, "6m": 0.3},
        )

        context = ScorerEvaluationContext(quick_sample_size=2)
        context.evaluate_all_targets()
        context.evaluate_all_targets()
        context.quick_evaluate(sample_size=2)
        context.quick_evaluate(sample_size=2)

        assert template_calls == [(0, 1, 2, 3), (0, 1)]


# ── Ground truth helper tests ───────────────────────────────────────────────


class TestGroundTruthHelpers:
    """Tests for forward return and snapshot date helpers."""

    def test_get_snapshot_dates(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _get_snapshot_dates

        # 3 years of daily dates
        dates = pd.date_range("2021-01-01", "2024-01-01", freq="B")  # business days
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": range(len(dates)),
        })
        snaps = _get_snapshot_dates(prices)
        assert len(snaps) > 0
        # All snapshots should be at least FORWARD_HORIZON_DAYS + 30 before end
        from valueinvestor.scorer_improver.ground_truth import FORWARD_HORIZON_DAYS
        from datetime import timedelta

        max_date = dates.max().date()
        for s in snaps:
            assert s <= max_date - timedelta(days=FORWARD_HORIZON_DAYS + 30)

    def test_get_snapshot_dates_short_data(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _get_snapshot_dates

        # Too short for any snapshots
        dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": range(len(dates)),
        })
        snaps = _get_snapshot_dates(prices)
        assert len(snaps) == 0

    def test_compute_forward_returns(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        # Create 2 years of daily prices for one stock
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="B")
        prices = pd.DataFrame({
            "ticker": "TEST",
            "date": dates,
            "close": [100.0 + i * 0.1 for i in range(len(dates))],
        })
        returns = _compute_forward_returns(prices, horizon_days=126)
        assert len(returns) > 0
        # Forward returns should be positive (monotonically increasing price)
        assert (returns["forward_return_6m"] > 0).all()

    def test_compute_forward_returns_uses_nearest_future_price(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        prices = pd.DataFrame({
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-30", "2024-02-02", "2024-03-20"]),
            "close": [100.0, 130.0, 140.0, 200.0],
        })

        returns = _compute_forward_returns(prices, horizon_days=30).set_index("date")

        assert returns.loc[pd.Timestamp("2024-01-01").date(), "forward_return_6m"] == 0.3
        assert pd.Timestamp("2024-02-02").date() not in returns.index

    def test_compute_forward_returns_empty(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        result = _compute_forward_returns(pd.DataFrame())
        assert result.empty

    def test_build_snapshot_features_vectorized_nearest_price(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _build_snapshot_features

        prices = pd.DataFrame({
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "date": pd.to_datetime([
                "2024-01-08",
                "2024-01-11",
                "2024-01-02",
                "2024-01-14",
                "2023-12-01",
            ]),
            "close": [10.0, 11.0, 20.0, 22.0, 30.0],
        })
        prices["date_dt"] = pd.to_datetime(prices["date"])
        valuations = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "pe_ratio": [12.0, 18.0],
            "pb_ratio": [1.2, 2.0],
        })
        financials = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "roe": [0.15, 0.20],
            "net_margin": [0.08, 0.10],
        })

        features = _build_snapshot_features(
            pd.Timestamp("2024-01-10").date(),
            prices,
            valuations,
            financials,
        ).set_index("ticker")

        assert set(features.index) == {"AAA", "BBB"}
        assert features.loc["AAA", "close"] == 11.0
        assert features.loc["BBB", "close"] == 22.0
        assert features.loc["AAA", "pe_ratio"] == 12.0
        assert features.loc["BBB", "roe"] == 0.20
