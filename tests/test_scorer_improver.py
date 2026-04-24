"""Tests for the scorer_improver sub-package.

Tests cover experiment_log, evaluator, agent, and ground_truth helpers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


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

        with patch.object(evaluator.importlib, "reload", side_effect=lambda mod: mod):
            with patch("valueinvestor.screener.scorer.MultiFactorScorer", return_value=RecordingScorer()):
                evaluator.evaluate_scorer(use_original_scores=False)

        assert seen["valuation"].ps_ratio is not None
        assert seen["valuation"].peg_ratio is not None
        assert seen["valuation"].dividend_yield is not None
        assert seen["financials"].revenue is not None
        assert seen["financials"].gross_margin is not None
        assert seen["financials"].operating_cash_flow is not None


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

    def test_compute_forward_returns_empty(self) -> None:
        from valueinvestor.scorer_improver.ground_truth import _compute_forward_returns

        result = _compute_forward_returns(pd.DataFrame())
        assert result.empty
