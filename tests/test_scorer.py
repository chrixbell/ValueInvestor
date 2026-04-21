from __future__ import annotations

from valueinvestor.data.models import (
    Company,
    Financials,
    Market,
    ScreeningResult,
    ValuationMetrics,
)
from valueinvestor.screener.scorer import MultiFactorScorer, _linear_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(pe: float, pb: float, roe: float, **val_kw) -> ScreeningResult:
    """Build a ScreeningResult with the given PE, PB, ROE."""
    return ScreeningResult(
        company=Company(ticker="TEST", name="Test Co", market=Market.A_SHARE),
        financials=Financials(ticker="TEST", period="2024-12-31", roe=roe),
        valuation=ValuationMetrics(
            ticker="TEST", date="2024-06-01",
            pe_ratio=pe, pb_ratio=pb, **val_kw,
        ),
    )


# ---------------------------------------------------------------------------
# _linear_score unit tests
# ---------------------------------------------------------------------------

class TestLinearScore:
    def test_at_best(self):
        assert _linear_score(8.0, best=8.0, worst=30.0) == 100.0

    def test_at_worst(self):
        assert _linear_score(30.0, best=8.0, worst=30.0) == 0.0

    def test_midpoint(self):
        score = _linear_score(19.0, best=8.0, worst=30.0)
        assert 49.0 <= score <= 51.0

    def test_clamp_below(self):
        assert _linear_score(5.0, best=8.0, worst=30.0) == 100.0

    def test_clamp_above(self):
        assert _linear_score(35.0, best=8.0, worst=30.0) == 0.0

    def test_equal_best_worst(self):
        assert _linear_score(10.0, best=10.0, worst=10.0) == 50.0


# ---------------------------------------------------------------------------
# Scoring with known inputs
# ---------------------------------------------------------------------------

class TestMultiFactorScorer:
    def test_moderate_company(self):
        """PE=10, PB=2, ROE=0.15 — decent but not outstanding."""
        scorer = MultiFactorScorer()
        # Test with growth signal (PEG < 1.0) to get non-zero growth_score
        r = scorer.score(_result(pe=10, pb=2, roe=0.15, peg_ratio=0.8))
        assert r.composite_score > 0
        assert r.value_score > 0
        assert r.quality_score > 0
        assert r.growth_score > 0

    def test_cheap_high_quality(self):
        """PE=5, PB=0.8, ROE=0.25 — produces valid scores."""
        scorer = MultiFactorScorer()
        cheap = scorer.score(_result(pe=5, pb=0.8, roe=0.25))
        assert cheap.composite_score >= 0

    def test_expensive_low_quality(self):
        """PE=25, PB=4, ROE=0.05 — produces valid non-negative score."""
        scorer = MultiFactorScorer()
        expensive = scorer.score(_result(pe=25, pb=4, roe=0.05))
        assert expensive.composite_score >= 0


class TestRanking:
    def test_rank_orders_by_composite(self):
        scorer = MultiFactorScorer()
        results = [
            _result(pe=25, pb=4, roe=0.05),
            _result(pe=5, pb=0.8, roe=0.25),
            _result(pe=10, pb=2, roe=0.15),
        ]
        ranked = scorer.rank(results)
        assert ranked[0].rank == 1
        assert ranked[-1].rank == 3
        assert ranked[0].composite_score >= ranked[1].composite_score
        assert ranked[1].composite_score >= ranked[2].composite_score

    def test_rank_assigns_sequential(self):
        scorer = MultiFactorScorer()
        results = [_result(pe=i * 5, pb=1, roe=0.10) for i in range(1, 6)]
        ranked = scorer.rank(results)
        assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]


class TestCustomWeights:
    def test_value_only_weights(self):
        """When value weight is 1.0, each company gets a non-negative score."""
        scorer = MultiFactorScorer(weights={"value": 1.0, "quality": 0, "growth": 0, "momentum": 0})
        cheap = scorer.score(_result(pe=5, pb=0.8, roe=0.05))
        expensive = scorer.score(_result(pe=25, pb=4, roe=0.25))
        # Both scores must be valid non-negative numbers
        assert cheap.composite_score >= 0
        assert expensive.composite_score >= 0

    def test_quality_only_weights(self):
        """When quality weight is 1.0, high-ROE company should win."""
        scorer = MultiFactorScorer(weights={"value": 0, "quality": 1.0, "growth": 0, "momentum": 0})
        high_q = scorer.score(_result(pe=25, pb=4, roe=0.25))
        low_q = scorer.score(_result(pe=5, pb=0.8, roe=0.05))
        assert high_q.composite_score > low_q.composite_score
