"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture stability
    "growth": 0.25,    # Increased to capture expansion potential
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # but allow the zero to propagate.
            normalized_val = max(score_val, 0.0) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of (S/100)^w is already the geometric mean when weights sum to 1.
            # If they don't sum to 1, we normalize by the total weight in the exponent.
            # However, (S/100)^w is actually S^(w) / 100^w. 
            # To get a score in [0, 100], we use the weighted power structure.
            # If total_weight is not 1, product_of_powers = prod( (S/100)^w ).
            # To scale back: composite_score = product_of_powers^(1/total_weight) * 100
            # But we need to handle the case where normalized_val is 0.
            if product_of_powers > 0:
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            else:
                composite_score = 0.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Use different metrics as fallbacks
        pe_f = result.valuation.pe_forward
        dividend = result.valuation.dividend_yield

        score_pe = 0.0
        if pe is not None and pe > 0:
            score_pe = _linear_score(1/pe, 1/50, 1/2) # mapping low PE to high score
        elif pe_f is not None and pe_f > 0:
            score_pe = _linear_score(1/pe_f, 1/50, 1/2)
        elif dividend is not None:
            score_pe = _linear_score(dividend, 0.02, 0.10)
        else:
            score_pe = 50.0

        score_pb = 0.0
        if pb is not None and pb > 0:
            score_pb = _linear_score(1/pb, 1/10, 1/0.5)
        else:
            score_pb = 50.0

        return (score_pe * 0.7) + (score_pb * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        
        # ROE score: high is good
        roe_score = _linear_score(roe, 0.05, 0.30)
        # Margin score: high is good
        margin_score = _linear_score(margin, 0.05, 0.25)
        # Debt score: low is good (clamped to avoid negative/weirdness)
        debt_score = _linear_score(1.0/max(debt_eq, 0.01), 1/2.0, 1/0.1)

        return (roe_score * 0.4) + (margin_score * 0.3) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio

        # ROE is a proxy for internal growth/efficiency
        roe_score = _linear_score(roe, 0.10, 0.40)
        
        # PEG score: low is better (but handle None/zero)
        if peg is not None and peg > 0:
            peg_score = _linear_score(1/peg, 1/3.0, 1/0.5)
        else:
            peg_score = 50.0

        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results