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
    "quality": 0.25,   # Increased to capture more stable returns
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
            # Use a small epsilon to avoid log(0) issues in geometric mean calculation
            # and ensure score_val/100 is strictly positive for the power operation.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product_of_powers already accounts for weights. 
            # If we want the weighted geometric mean (S1^w1 * S2^w2...)^(1/sum_w),
            # we apply the exponent to the product.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculates value score using PE and PB ratios."""
        # Using a combination of PE and PB for valuation.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback values for clamping/bounds
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        v_pe = 0.0
        if pe is not None and pe > 0:
            # Lower PE is better. We invert the logic for _linear_score if needed, 
            # but here we just define best/worst such that low PE = high score.
            v_pe = _linear_score(pe, pe_worst, pe_best)
        
        v_pb = 0.0
        if pb is not None and pb > 0:
            v_pb = _linear_score(pb, pb_worst, pb_best)

        # If data is missing, we rely on the available metric.
        if pe is not None and pb is not None:
            return (v_pe + v_pb) / 2.0
        elif pe is not None:
            return v_pe
        elif pb is not None:
            return v_pb
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE: Higher is better
        q_roe = 0.0
        if roe is not None:
            # Scale ROE linearly (assuming -20% to 40% range)
            q_roe = _linear_score(roe * 100, 40.0, -20.0)

        # Debt/Equity: Lower is better
        q_debt = 0.0
        if debt_equity is not None:
            # Scale Debt/Equity (assuming 0 to 2.0 range)
            q_debt = _linear_score(debt_equity, 2.0, 0.0)
        elif debt_equity is None:
            # If no debt info, assume neutral-to-good if we can't penalize
            q_debt = 50.0

        if roe is not None and debt_equity is not None:
            return (q_roe + q_debt) / 2.0
        elif roe is not None:
            return q_roe
        elif debt_equity is not None:
            return q_debt
        return 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using PEG ratio."""
        peg = result.valuation.peg_ratio
        
        if peg is not None and peg > 0:
            # Lower PEG is better (growth at reasonable price)
            return _linear_score(peg, 0.5, 3.0)
        elif peg is None:
            return 50.0
        else:
            # If PEG is zero or negative due to negative earnings, it's tricky.
            # For now, return a neutral score.
            return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results