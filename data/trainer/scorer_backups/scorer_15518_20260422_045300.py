"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased to prioritize stable business models
    "growth": 0.25,    # Balanced with quality for a more holistic approach
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

        # We use an additive weighted average for the composite score instead of 
        # geometric mean to prevent a single zero-score (e.g., from one missing 
        # metric) from wiping out the entire composite score, while still 
        # allowing individual factor influence.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use PE if available, else PB. 
        # Low PE/PB is better for value.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Thresholds for clamping
        pe_best, pe_worst = 5.0, 40.0
        pb_best, pb_worst = 0.5, 5.0

        if pe is not None and pe > 0:
            # If PE is provided, we use it as the primary value metric.
            # We map lower PE to higher scores.
            return _linear_score(pe, pe_best, pe_worst) if pe < pe_best else (100.0 - _linear_score(pe, pe_best, pe_worst))
        elif pb is not None and pb > 0:
            return _linear_score(pb, pb_best, pb_worst) if pb < pb_best else (100.0 - _linear_score(pb, pb_best, pb_worst))
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity

        # Higher ROE is better, lower Debt/Equity is better.
        roe_score = 50.0
        if roe is not None:
            # Map ROE (assume -20% to 40% range)
            roe_score = _linear_score(roe * 100, 5.0, 30.0)

        debt_score = 50.0
        if debt_eq is not None:
            # Map Debt/Equity (assume 0 to 2.0 range)
            debt_score = _linear_score(debt_eq, 0.1, 1.5)

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a growth-adjusted valuation metric. Lower is better.
        peg_score = 50.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.0)
        elif peg is None:
            # Fallback to ROE if PEG is unavailable
            if roe is not None:
                peg_score = _linear_score(roe * 100, 5.0, 25.0)
            else:
                peg_score = 50.0

        return peg_score

    def _rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results