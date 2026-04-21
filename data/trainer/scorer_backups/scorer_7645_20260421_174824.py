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
    "quality": 0.25,   # Increased to prioritize stability and profitability
    "growth": 0.25,    # Balanced with quality to capture high-return profiles
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
            # Use a small epsilon to prevent math domain errors if score is 0
            # but allow the factor to zero out the geometric mean.
            normalized_score = score_val / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula (product^(1/total_weight)) is applied to the 
            # product of weighted scores. Since we are multiplying (S/100)^w, 
            # the product is already scaled. We just need to ensure it's normalized.
            # To keep the result in [0, 100], we use the product as is.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Value scoring using PE and PB ratios."""
        # Using PEG as a secondary check for value-growth trade-off
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Handle None values: if data is missing, we assign a neutral but cautious score
        if pe is None or pb is None:
            # Fallback to PB if PE is missing, or vice versa. 
            # If both are None, return a neutral score.
            if pe is None and pb is None:
                return 50.0
            val = pe if pe is not None else pb
            # Use a placeholder logic for missingness
            return _linear_score(val if val > 0 else 1, 20, 50)

        # Basic value score: lower PE and PB are better.
        # We use a wide range to avoid extreme sensitivity.
        score_pe = _linear_score(pe, 5, 40) if pe > 0 else 0.0
        score_pb = _linear_score(pb, 1, 10) if pb > 0 else 0.0
        
        # PEG inclusion: lower is better (value-growth balance)
        score_peg = 50.0
        if peg is not None and peg > 0:
            score_peg = _linear_score(peg, 0.5, 3.0)

        return (score_pe * 0.4 + score_pb * 0.4 + score_peg * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Quality scoring using ROE and Leverage."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        net_margin = result.financials.net_margin

        if roe is None or debt_to_equity is None:
            return 50.0

        # Higher ROE is better
        score_roe = _linear_score(roe, 5.0, 25.0)
        
        # Lower Debt-to-Equity is better (assuming 1.0 is neutral)
        # We reverse the scale: higher debt -> lower score
        score_debt = _linear_score(1/max(debt_to_equity, 0.01), 0.1, 2.0) if debt_to_equity and debt_to_equity > 0 else 50.0
        if debt_to_equity == 0: score_debt = 100.0

        # Margin stability
        score_margin = _linear_score(net_margin if net_margin is not None else 10.0, 5.0, 30.0)

        return (score_roe * 0.5 + score_debt * 0.3 + score_margin * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Growth scoring using ROE and potential."""
        roe = result.financials.roe
        # In this context, we use ROE as a proxy for growth-driven quality
        if roe is None:
            return 50.0
        
        # Growth focus: slightly higher ROE thresholds
        return _linear_score(roe, 10.0, 35.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results