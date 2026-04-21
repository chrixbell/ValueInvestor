"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to give more room to quality/growth
    "quality": 0.25,   # Increased to capture more stability
    "growth": 0.25,    # Increased to capture alpha in growth-oriented markets
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

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # We use a weighted arithmetic mean of the scores as a baseline.
        # In many financial applications, if one factor is 0 (e.g., a company with no revenue),
        # the geometric mean might be too punitive for ranking purposes.
        # However, we will use a slightly adjusted approach to ensure stability.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # We use a combination of PE and PB to capture different valuation aspects.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle potential None or zero/negative values for PE/PB
        pe = pe if pe is not None and pe > 0 else 25.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Use a simple ranking-friendly linear score for valuation
        # We consider PE up to 30 and PB up to 5 as reasonable bounds.
        # If PE is higher than 30, it gets a very low score.
        score_pe = _linear_score(pe, 5.0, 30.0) # 5 is best, 30 is worst
        score_pb = _linear_score(pb, 1.0, 5.0)  # 1 is best, 5 is worst
        
        return (score_pe + score_pb) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE: Assume a range of -10% to 30% is relevant.
        # We map it so that higher ROE = higher score.
        score_roe = _linear_score(roe * 100, 25.0, -10.0)
        
        # Debt to Equity: Lower is better. Assume 0% to 150% range.
        # If debt_to_equity is negative (unlikely), treat as 0.
        de = max(0.0, debt_equity)
        score_debt = _linear_score(de, 0.0, 1.5) # 0 is best, 1.5 (150%) is worst
        
        return (score_roe + score_debt) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG: Lower is better (growth at a reasonable price). 
        # A PEG of 0.5 is great, 2.0 is mediocre, 4.0 is expensive.
        # We use log-scale logic via linear mapping to handle the range.
        score_peg = _linear_score(peg, 0.5, 3.0)
        
        # ROE as a proxy for growth quality (secondary check)
        score_roe = _linear_score(roe * 100, 20.0, -5.0)
        
        return (score_peg + score_roe) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort all results by composite score and assign ranks."""
        # Sort descending (higher score = better rank)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results