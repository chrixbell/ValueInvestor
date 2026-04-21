"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture more stable returns
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
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
            # Use a small epsilon to prevent math domain errors with 0.0 score
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Value scoring using PE and PB ratios."""
        # We use a combination of PE and PB. 
        # Lower is better for both.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Fallback values for clamping
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        # If PE is negative (common in China), we treat it as "worst" or use a very high PE
        if pe is None or pe <= 0:
            pe_val = pe_worst
        else:
            pe_val = pe

        if pb is None or pb <= 0:
            pb_val = pb_worst
        else:
            pb_val = pb

        # Inverse linear score (since lower is better)
        # We transform to a "higher is better" scale first
        pe_score = (pe_worst - pe_val) / (pe_worst - pe_best + 1e-6) * 100.0
        pb_score = (pb_worst - pb_val) / (pb_worst - pb_best + 1e-6) * 100.0
        
        # Clamp and average
        pe_score = max(0.0, min(100.0, pe_score))
        pb_score = max(0.0, min(100.0, pb_score))
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Quality scoring using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # Higher ROE is better
        roe_best, roe_worst = 0.25, 0.0
        # Lower Debt/Equity is better
        de_best, de_worst = 0.2, 1.5

        roe_score = (roe - roe_worst) / (roe_best - roe_worst + 1e-6) * 100.0
        de_score = (de_worst - debt_equity) / (de_worst - de_best + 1e-6) * 100.0

        roe_score = max(0.0, min(100.0, roe_score))
        de_score = max(0.0, min(100.0, de_score))

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Growth scoring using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0

        # Higher ROE is better for growth potential
        roe_best, roe_worst = 0.25, 0.0
        # Lower PEG is better (growth at reasonable price)
        peg_best, peg_worst = 0.5, 3.0

        roe_score = (roe - roe_worst) / (roe_best - roe_worst + 1e-6) * 100.0
        peg_score = (peg_worst - peg) / (peg_worst - peg_best + 1e-6) * 100.0

        roe_score = max(0.0, min(100.0, roe_score))
        peg_score = max(0.0, min(100.0, peg_score))

        return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results