"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for valuation
    "quality": 0.25,   # Increased quality to capture stable earners
    "growth": 0.25,    # Balanced with quality for sustainable growth
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

    def score(self, result: Screening_Result) -> Screening_Result:
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
            # Use a small epsilon to prevent zero-multiplication in geometric mean
            # while maintaining the ability for bad stocks to score low.
            safe_score = max(0.001, score_val) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # We use the weighted product directly. Since it's (S/100)^W, 
            # multiplying by 100 at the end gives the scaled composite.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use PE as primary, fallback to PB or PS
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Define ranges for linear scoring
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.5, 6.0

        # Handle negative PE (loss-making)
        if pe is None or pe <= 0:
            # If PE is invalid/negative, use PB if available, else low score
            if pb and pb > 0:
                return _linear_score(pb, pb_best, pb_worst)
            return 0.0

        # Calculate score based on PE
        v_score = _linear_score(pe, pe_best, pe_worst)

        # If PB is available and provides a significantly better/worse signal, blend it
        if pb and pb > 0:
            pb_score = _linear_score(pb, pb_best, pb_worst)
            v_score = (v_score + pb_score) / 2.0

        return v_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity

        # ROE is a primary quality metric
        if roe is None:
            return 50.0

        # We want higher ROE to be better.
        roe_best, roe_worst = 0.20, 0.05
        q_score = _linear_score(roe, roe_best, roe_worst)

        # Debt/Equity constraint
        if debt_to_equity is not None:
            # High debt reduces quality score
            d_best, d_worst = 0.3, 1.5
            # For debt, lower is better, so we swap best/worst in linear_score logic
            # or simply invert the result. 
            # Actually, we'll use a custom logic:
            d_score = max(0.0, min(100.0, (1.5 - debt_to_equity) / (1.5 - 0.3) * 100.0)) if debt_to_equity < 1.5 else 0.0
            # Blend ROE score with debt score (weighted towards ROE)
            q_score = q_score * 0.7 + d_score * 0.3

        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # If PEG is available, it's a strong growth/valuation hybrid
        if peg is not None and peg > 0:
            peg_best, peg_worst = 0.5, 2.5
            g_score = _linear_score(peg, peg_best, peg_worst)
            return g_score

        # Fallback to ROE-based growth if PEG is not available
        if roe is not None:
            return _linear_score(roe, 0.15, 0.05)
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results