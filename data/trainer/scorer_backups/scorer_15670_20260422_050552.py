"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slight reduction to allow more breathing room for quality/growth
    "quality": 0.25,   # Increased quality weight to penalize bad balance sheets
    "growth": 0.25,    # Balanced with quality to capture growth-at-a-reasonable-price
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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

        # Momentum is a placeholder — set to 50 (neutral).
        momentum_score = 50.0 

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # We use a modified geometric mean logic. 
        # To ensure that a zero in one category doesn't wipe out everything (unless intended),
        # we use an epsilon-offset for the geometric mean calculation.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] range. We add epsilon to handle zero scores in geometric mean.
            normalized_score = (max(0.0, score_val) + epsilon) / (100.0 + epsilon)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers is already effectively (S1/100)^w1 * (S2/100)^w2...
            # Because we used normalized_score = (val + eps)/(100 + eps), the result is naturally in [0, 1].
            # We scale back to 100.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        v = result.valuation
        if not v or (v.pe_ratio is None and v.pb_ratio is None):
            return 50.0

        # Prioritize PE if available, then PB. 
        # Use a range that captures reasonable valuation-to-growth levels.
        if v.pe_ratio is not None and v.pe_ratio > 0:
            # PE-based score (Lower is better)
            # Using a range of 0.5 to 30 for PE
            score = _linear_score(v.pe_ratio, 2.0, 30.0)
        elif v.pb_ratio is not None and v.pb_ratio > 0:
            # PB-based score (Lower is better)
            score = _linear_score(v.pb_ratio, 0.5, 6.0)
        else:
            score = 50.0
        return score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        f = result.financials
        if not f:
            return 50.0

        # ROE is a primary quality driver
        roe = f.roe if f.roe is not None else 0.0
        # Debt/Equity: lower is better
        de = f.debt_to_equity if f.debt_to_equity is not None else 0.5

        # ROE score (Higher is better)
        roe_score = _linear_score(roe, 5.0, 25.0)
        # Debt score (Lower is better)
        de_score = _linear_score(de, 0.1, 1.5)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        # PEG ratio (Lower is better)
        peg = v.peg_ratio if v.peg_ratio is not None and v.peg_ratio > 0 else None
        # ROE (Higher is better)
        roe = f.roe if f.roe is not None else 0.0

        if peg is not None:
            # PEG-based score (Lower is better)
            score = _linear_score(peg, 0.5, 2.0)
        else:
            # Fallback to ROE if PEG is unavailable
            score = _linear_score(roe, 5.0, 25.0)
        
        return score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            # Ensure scores are calculated before ranking
            if not hasattr(r, 'composite_score'):
                self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results