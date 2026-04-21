"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.55,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to reward stability and profitability
    "growth": 0.20,    # Maintain growth component
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # but keep the impact of a zero score significant.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula with weights is (Product(S_i^w_i))^(1/Sum(w_i))
            # If we want the result in [0, 100], we scale it back.
            # Note: if any score is 0, product_of_powers becomes 0.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use PEG as a tie-breaker/supplement if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Common reasonable bounds for A-share/HK stocks
        pe_best, pe_worst = 15.0, 60.0
        pb_best, pb_worst = 1.5, 8.0

        v_score = 0.0
        count = 0

        if pe is not None and pe > 0:
            v_score += _linear_score(pe, pe_best, pe_worst)
            count += 1
        elif pe is not None and pe <= 0: # Negative PE handled as worst case
            v_score += 0.0
            count += 1

        if pb is not None:
            v_score += _linear_score(pb, pb_best, pb_worst)
            count += 1

        # If we have PE/PB, normalize the average. Otherwise if only PEG:
        if count > 0:
            v_score = v_score / count
        elif peg is not None and peg > 0:
            v_score = _linear_score(peg, 0.5, 3.0)
        else:
            v_score = 50.0

        return v_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        q_score = 0.0
        count = 0

        if roe is not None:
            # ROE target: 15% to 30%+
            q_score += _linear_score(roe, 15.0, 35.0)
            count += 1
        
        if debt_equity is not None:
            # Lower debt to equity is better. 
            # Map higher debt (e.g. 2.0) to 0 and lower (e.g. 0.2) to 100
            q_score += _linear_score(debt_equity, 0.2, 2.0)
            count += 1

        if count > 0:
            return q_score / count
        return 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        g_score = 0.0
        count = 0

        if roe is not None:
            g_score += _linear_score(roe, 10.0, 30.0)
            count += 1
        
        if peg is not None and peg > 0:
            # Lower PEG is better for growth-value balance
            g_score += _linear_score(peg, 0.5, 2.0)
            count += 1

        if count > 0:
            return g_score / count
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results