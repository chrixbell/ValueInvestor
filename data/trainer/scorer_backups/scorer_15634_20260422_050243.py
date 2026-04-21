"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced value to allow more room for quality/growth
    "quality": 0.25,   # Increased quality to capture more stable returns
    "growth": 0.25,    # Increased growth to capture upside potential
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

        # Use an arithmetic mean of normalized scores for better stability 
        # in cases where one factor might be zero.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] and add to sum
            weighted_sum += (score_val / 100.0) * weight

        if total_weight > 0:
            # Result is scaled back to [0, 100]
            composite_score = (weighted_sum / total_weight) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Default values for clamping if data is missing
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        score_pe = _linear_score(pe if pe is not None else pe_worst, pe_best, pe_worst) if pe is not None else 50.0
        score_pb = _linear_score(pb, pb_best, pb_worst) if pb is not None else 50.0
        
        # If PE/PB are extremely high, they map to 0 via clamping.
        return (score_pe + score_pb) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE: higher is better
        if roe is not None:
            score_roe = _linear_score(roe, 15.0, 30.0) # 15% is best-ish, 30% is very high
        else:
            score_roe = 50.0

        # Debt/Equity: lower is better
        if debt_equity is not None:
            # A debt-to-equity of 0.5 is "best" (100), 2.0 is "worst" (0)
            score_debt = _linear_score(debt_equity, 0.5, 2.0)
        else:
            score_debt = 50.0

        return (score_roe + score_debt) / 2.0

    def _growth_score(self, result: Screening_Result) -> float:
        """Compute growth score using ROE and PEG."""
        # Re-using the logic for growth/return potential
        roe = result.financials.roe if result.financials.roe is not None else None
        peg = result.valuation.peg_ratio

        if roe is not None:
            score_roe = _linear_score(roe, 10.0, 25.0)
        else:
            score_roe = 50.0

        if peg is not None and peg > 0:
            # PEG: lower is better (growth at a reasonable price)
            score_peg = _linear_score(peg, 0.5, 2.0)
        else:
            score_peg = 50.0

        return (score_roe + score_peg) / 2.0