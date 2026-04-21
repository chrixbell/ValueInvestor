"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Focused on valuation efficiency
    "quality": 0.30,   # High quality is a strong predictor of stability
    "growth": 0.20,    # Growth as a secondary driver
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Using a small epsilon to prevent math domain errors with 0 scores in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # We use the direct product of powers as it is already scaled correctly by weights.
            # (S1^w1 * S2^w2 ...) where sum(weights) = 1
            # If total_weight != 1, we adjust.
            composite_score = product_of_powers ** (1.0 / total_weight) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE if available, else trailing PE. 
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Penalty for negative PE (often indicates loss)
        if pe is not None and pe <= 0:
            pe_score = 0.0
        elif pe is not None and pb is not None:
            # Combine PE and PB into a single value-based score
            # We use linear interpolation with reasonable bounds for A-share/HK markets
            pe_score = _linear_score(pe, 10.0, 40.0)
            pb_score = _linear_score(pb, 1.0, 5.0)
            pe_score = (pe_score + pb_score) / 2.0
        elif pe is not None:
            pe_score = _linear_score(pe, 10.0, 40.0)
        else:
            pe_score = 50.0
            
        return pe_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        if roe is not None and debt_equity is not None:
            # Higher ROE is good, lower Debt/Equity is good.
            roe_score = _linear_score(roe, 5.0, 25.0)
            # For debt_to_equity, lower is better, so we flip the logic for linear_score
            # If debt/equity is 0.5, it's better than 2.0.
            # We can simulate this by passing 'best' as the lower value in a custom logic or just transform.
            # Let's use: score = 100 * (worst_debt / current_debt) clamped.
            # Simplified: 
            de_score = 100.0 if debt_equity <= 0.5 else (max(0.0, 100.0 - (debt_equity * 20)))
            return (roe_score + de_score) / 2.0
        elif roe is not None:
            return _linear_score(roe, 5.0, 25.0)
        elif debt_equity is not None:
            return 100.0 if debt_equity <= 0.5 else (max(0.0, 100.0 - (debt_equity * 20)))
        else:
            return 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG ratio."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        if peg is not None and peg > 0:
            # Lower PEG is better (growth at a reasonable price)
            return _linear_score(1/peg, 0.5, 3.0) # Inverting to use linear_score logic
        elif roe is not None:
            return _linear_score(roe, 5.0, 25.0)
        else:
            return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results