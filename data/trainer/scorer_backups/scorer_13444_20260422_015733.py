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
    "quality": 0.25,   # Increased to capture stable earners
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
            # Use a small epsilon to prevent log(0) issues if score_val is 0
            # but allow the zero to significantly penalize the product.
            norm_score = max(0.0, score_val) / 100.0
            product_of_powers *= (norm_score ** weight)

        if total_weight > 0:
            # Geometric mean approach for multi-factor scoring
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Using a mix of PE and PB for valuation. 
        # We use logical bounds to prevent extreme outliers from dominating.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        score_pe = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 25] for high score.
            score_pe = _linear_score(pe, 1.0, 30.0)
            # Invert because lower PE is better
            score_pe = 100.0 - score_pe
        elif pe is not None and pe <= 0:
            score_pe = 100.0 # Negative PE (profitable) is often very good

        score_pb = 0.0
        if pb is not None and pb > 0:
            # Target PB range [0.5, 3] for high score. 
            score_pb = _linear_score(pb, 0.5, 5.0)
            score_pb = 100.0 - score_pb
        elif pb is not None and pb <= 0:
            score_pb = 100.0

        # Weighted average of PE and PB
        if pe is not None and pb is not None:
            return (score_pe + score_pb) / 2.0
        return score_pe if pe is not None else score_pb

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and debt levels."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        score_roe = 0.0
        if roe is not None:
            # Higher ROE is better. Use linear scaling for typical ranges.
            score_roe = _linear_score(roe, 15.0, -5.0)

        score_debt = 0.0
        if debt_equity is not None:
            # Lower debt-to-equity is better.
            score_debt = _linear_score(debt_equity, 0.2, 1.5)
            score_debt = 100.0 - score_debt

        if roe is not None and debt_equity is not None:
            return (score_roe + score_debt) / 2.0
        return score_roe if roe is not None else score_debt

    def _growth_score(self, result: Screening_Result) -> float:
        """Calculate growth score based on ROE and PEG."""
        # Note: Re-using ROE in growth context as a proxy for capital efficiency/growth capability
        # or using peg_ratio if available.
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        score_peg = 0.0
        if peg is not None and peg > 0:
            # Lower PEG is better. Target range [0.5, 2.0]
            score_peg = _linear_score(peg, 0.5, 2.5)
            score_peg = 100.0 - score_peg
        elif peg is not None and peg <= 0:
            score_peg = 100.0

        if peg is not None and roe is not None:
            # We use PEG as the primary growth signal here. 
            return score_peg
        elif peg is not None:
            return score_peg
        elif roe is not None:
            # Fallback to ROE scaled for growth-like values
            return _linear_score(roe, 20.0, -10.0)
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