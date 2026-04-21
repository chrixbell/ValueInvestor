"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture stable earnings power
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

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Using a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # We treat a 0 score as a very small number so it doesn't zero out the whole product
            # but still heavily penalizes the stock.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean.
            # Since we used (score/100)^weight, we scale back by 100.
            # We use (total_weight) to normalize the exponent if weights don't sum to 1.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        v = result.valuation
        if not v:
            return 50.0

        # Primary value drivers
        pe_score = 0.0
        if v.pe_ratio is not None and v.pe_ratio > 0:
            # Use a reasonable range for PE (1 to 30)
            pe_score = _linear_score(v.pe_ratio, 1.0, 30.0)
            # Invert because lower PE is better
            pe_score = 100.0 - pe_score
        elif v.pe_ratio is not None and v.pe_ratio <= 0:
            # Negative PE is tricky, but logically it's often "cheap" (though risky)
            pe_score = 70.0 
        else:
            pe_score = 50.0

        pb_score = 0.0
        if v.pb_ratio is not None and v.pb_ratio > 0:
            pb_score = 100.0 - _linear_score(v.pb_ratio, 0.5, 10.0)
        elif v.pb_ratio is not None:
            pb_score = 50.0
        else:
            pb_score = 50.0

        # Combine PE and PB (weighted towards PE)
        return (pe_score * 0.7) + (pb_score * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        f = result.financials
        if not f:
            return 50.0

        # ROE component (Higher is better)
        roe_score = 50.0
        if f.roe is not None:
            # Map ROE (e.g., -20% to 40%) to [0, 100]
            roe_score = _linear_score(f.roe * 100, -10.0, 30.0)

        # Leverage component (Lower is better)
        leverage_score = 50.0
        if f.debt_to_equity is not None:
            # Map Debt/Equity (e.g., 0 to 2.0) to [100, 0]
            leverage_score = 100.0 - _linear_score(f.debt_to_equity, 0.0, 2.0)
        elif f.debt_to_equity is None:
            leverage_score = 50.0

        return (roe_score * 0.6) + (leverage_score * 0.4)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        # PEG component (Lower is better)
        peg_score = 50.0
        if v.peg_ratio is not None and v.peg_ratio > 0:
            peg_score = 100.0 - _linear_score(v.peg_ratio, 0.5, 3.0)
        elif v.peg_ratio is not None:
            peg_score = 50.0

        # ROE as a proxy for growth/efficiency (Higher is better)
        roe_growth_score = 50.0
        if f.roe is not None:
            roe_growth_score = _linear_score(f.roe * 100, -5.0, 25.0)

        return (peg_score * 0.4) + (roe_growth_score * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results