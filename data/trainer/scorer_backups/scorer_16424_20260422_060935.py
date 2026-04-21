"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow quality/growth more room
    "quality": 0.25,   # Increased to emphasize robust balance sheets
    "growth": 0.25,    # Balanced with quality
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

        # Momentum is a placeholder
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
            # Use a small epsilon to prevent zero-multiplication issues in geometric mean
            # while still allowing low scores to pull the composite down.
            safe_score = max(0.001, score_val)
            product_of_powers *= (safe_score / 100.0) ** weight

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        v = result.valuation
        if not v:
            return 50.0

        # We prioritize PE but use PB as a secondary check
        pe = v.pe_ratio
        pb = v.pb_ratio

        # Use reasonable bounds for A-share/HK stocks
        pe_best, pe_worst = 15.0, 40.0
        pb_best, pb_worst = 1.5, 5.0

        pe_s = _linear_score(pe if pe is not None else 25.0, pe_best, pe_worst) if pe is not None else 50.0
        pb_s = _linear_score(pb if pb is not None else 2.5, pb_best, pb_worst) if pb is not None else 50.0
        
        # If PE is negative (loss making), it's usually bad for value, but we clamp to 0
        if pe is not None and pe <= 0:
            pe_s = 0.0

        return (pe_s * 0.7) + (pb_s * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        f = result.financials
        if not f:
            return 50.0

        roe = f.roe
        debt_equity = f.debt_to_equity

        # ROE is a primary quality driver
        roe_s = _linear_score(roe if roe is not None else 10.0, 20.0, 5.0) if roe is not None else 50.0
        # Lower debt is better: reverse linear score logic
        de_s = _linear_score(debt_equity if debt_equity is not None else 0.5, 0.2, 1.5) if debt_equity is not None else 50.0
        if debt_equity is not None and debt_equity <= 0: # No debt is perfect
            de_s = 100.0

        return (roe_s * 0.7) + (de_s * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using ROE and PEG."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        # Growth often correlates with ROE and PEG (Price/Earnings-to-Growth)
        roe = f.roe if f.roe is not None else 10.0
        peg = v.peg_ratio if v.peg_ratio is not None else 1.0

        roe_s = _linear_score(roe, 15.0, 5.0)
        # Lower PEG is better (growth relative to price)
        peg_s = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0
        if peg <= 0: # Negative PEG is tricky, but could imply high growth/low PE
            peg_s = 50.0

        return (roe_s * 0.5) + (peg_s * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results