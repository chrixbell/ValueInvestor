"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights – increased size weight, reduced growth weight
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,
    "quality": 0.30,
    "growth": 0.15,
    "momentum": 0.00,
    "size": 0.10,
}


def _linear_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Values beyond the endpoints are clamped.
    """
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Assumes 'value', 'best', and 'worst' are positive for math.log.
    Values beyond the endpoints are clamped.
    """
    if value <= 0 or best <= 0 or worst <= 0:
        return 0.0

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0

        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, growth, and size dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        # Momentum is a placeholder — set to 50 (neutral).
        momentum_score = 50.0

        # Size score – new
        size_score = self._size_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
            "size": size_score,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Weighted sum aggregation
        composite_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            composite_score += weight * score_val

        if total_weight > 0:
            composite_score /= total_weight
        else:
            composite_score = 50.0

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB → higher score, using geometric mean of individual scores."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        pe_score = _linear_score(1.0/pe, 1.0/50.0, 1.0/5.0) if pe > 0 else 0.0
        pb_score = _linear_score(1.0/pb, 1.0/10.0, 1.0/1.0) if pb > 0 else 0.0

        # Geometric mean – penalises asymmetry and ensures both dimensions are cheap
        return math.sqrt(pe_score * pb_score)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE, net margin, and FCF margin → higher score.
        Removed debt and current ratio to reduce noise, concentrating on profitability
        and cash generation quality.
        """
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        roe_score = _linear_score(roe, 30.0, -10.0)
        margin_score = _linear_score(margin, 20.0, -5.0)

        # Free cash flow margin score
        fcf = result.financials.free_cash_flow
        revenue = result.financials.revenue
        fcf_margin_score = 50.0
        if fcf is not None and revenue is not None and revenue > 0:
            fcf_margin = fcf / revenue
            fcf_margin_score = _linear_score(fcf_margin, 0.20, -0.10)
        elif fcf is not None and fcf > 0:
            fcf_margin_score = 60.0

        # New simplified weights: ROE 35%, margin 25%, FCF margin 40%
        return (roe_score * 0.35 +
                margin_score * 0.25 +
                fcf_margin_score * 0.40)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG → higher score.
        Note: ROE removed to reduce redundancy with quality sub-score.
        Growth now relies solely on PEG ratio.
        """
        peg = result.valuation.peg_ratio

        if peg is not None and peg > 0:
            peg_score = _linear_score(1.0/peg, 1.0/0.5, 1.0/3.0)
        else:
            peg_score = 50.0

        return peg_score

    def _size_score(self, result: ScreeningResult) -> float:
        """Smaller market cap → higher score (small-cap tilt)."""
        mcap = result.valuation.market_cap_rmb
        if mcap is None or mcap <= 0:
            return 50.0

        # Use log10 to linearise size distribution
        log_mcap = math.log10(mcap)
        # Best (smallest cap) ~ 1e8 (100M RMB), worst (largest) ~ 1e13 (10T RMB)
        log_best = math.log10(1e8)   # 100 million
        log_worst = math.log10(1e13) # 10 trillion
        score = _linear_score(log_mcap, log_best, log_worst)
        # Invert: smaller cap → higher score
        return 100.0 - score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)

        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)

        for i, res in enumerate(sorted_results):
            res.rank = i + 1

        return sorted_results