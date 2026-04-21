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
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Increased to capture upside potential
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
            # Use a small epsilon to prevent zero-multiplication errors in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Geometric mean calculation
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # If the result is effectively zero due to a 0 score, ensure it handles correctly
            if math.isnan(composite_score):
                composite_score = 0.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield score better."""
        # Use a mix of valuation metrics. 
        # Handle None by providing safe defaults to prevent crashes.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Score components
        v1 = _linear_score(10.0, 5.0, 30.0) if pe is not None else 50.0
        v2 = _linear_score(1.0, 0.5, 3.0) if pb is not None else 50.0
        v3 = _linear_score(4.0, 1.0, 8.0) if div is not None else 50.0
        
        # If PE is negative (loss making), it's often high risk but can be value. 
        # For this model, we treat very low/negative PE as a specific case.
        if pe is not None and pe <= 0:
            v1 = 10.0 # Low score for negative PE to avoid ranking loss-makers as top value

        return (v1 * 0.4 + v2 * 0.3 + v3 * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage score better."""
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        # ROE is a primary quality driver
        q1 = _linear_score(15.0, 5.0, 30.0) if roe is not None else 50.0
        # Margin stability
        q2 = _linear_score(10.0, 2.0, 20.0) if margin is not None else 50.0
        # Leverage (Lower is better)
        q3 = _linear_score(0.5, 0.0, 1.5) if debt is not None else 50.0
        # Flip q3 because lower debt should be higher score
        if debt is not None:
            q3 = 100.0 - q3

        return (q1 * 0.5 + q2 * 0.3 + q3 * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG score better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a great growth-at-reasonable-price metric
        g1 = _linear_score(1.0, 0.5, 2.0) if peg is not None else 50.0
        # ROE as a proxy for internal growth capacity
        g2 = _linear_score(20.0, 5.0, 40.0) if roe is not None else 50.0

        return (g1 * 0.6 + g2 * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks all results based on the composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results