"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more room for quality/growth
    "quality": 0.25,   # Increased to penalize companies with poor fundamentals
    "growth": 0.25,    # Increased to capture expansion potential
    "momentum": 0.00,  
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

    log_value = math.log(value)
    log_best = math.log(best)
    log_worst = math.log(worst)

    if log_best == log_worst:
        return 50.0
    
    score = (log_value - log_worst) / (log_best - log_worst) * 100.0
    return max(0.0, min(100.0, score))


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

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

        momentum_score = 50.0 

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # To prevent a single zero score from zeroing out the entire composite,
        # we use a small epsilon for the geometric mean calculation.
        epsilon = 0.01
        
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use epsilon to ensure a zero score doesn't destroy the whole product, 
            # but still penalizes heavily.
            normalized_score = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula is (Product of x_i^w_i) ^ (1 / Sum of w_i)
            # However, the product above is already effectively scaled by weights.
            # If weight sum is 1.0, result is just product_of_powers * 100.
            # We adjust to ensure the scale is correct.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a mix of valuation metrics
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        yield_val = result.valuation.dividend_yield

        # Fallback values for clamping
        pe_best, pe_worst = 10.0, 50.0
        pb_best, pb_worst = 1.0, 10.0
        ps_best, ps_worst = 1.0, 5.0
        yield_best, yield_worst = 5.0, 0.0

        scores = []
        if pe is not None and pe > 0:
            scores.append(_linear_score(pe, pe_best, pe_worst) if pe < pe_worst else 0.0)
        if pb is not None and pb > 0:
            scores.append(_linear_score(pb, pb_best, pb_worst) if pb < pb_worst else 0.0)
        if ps is not None and ps > 0:
            scores.append(_linear_score(ps, ps_best, ps_worst) if ps < ps_worst else 0.0)
        if yield_val is not None:
            scores.append(_linear_score(yield_val, yield_best, yield_worst))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        scores = []
        # ROE is a primary quality driver
        if roe is not None:
            scores.append(_linear_score(roe, 15.0, -5.0))
        # Leverage (Debt/Equity) - lower is better
        if debt_equity is not None:
            scores.append(_linear_score(debt_equity, 0.5, 2.0))
        # Net Margin
        if margin is not None:
            scores.append(_linear_score(margin, 10.0, -5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        scores = []
        if peg is not None and peg is not None and peg > 0:
            # Low PEG is high growth value
            scores.append(_linear_score(peg, 0.5, 2.0))
        if roe is not None:
            scores.append(_linear_score(roe, 10.0, -5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite_score."""
        # Sort by score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results