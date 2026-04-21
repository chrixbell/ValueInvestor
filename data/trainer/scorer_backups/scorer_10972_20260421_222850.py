"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more influence from quality/growth
    "quality": 0.25,   # Increased to penalize low-quality companies more effectively
    "growth": 0.25,    # Increased to capture potential upside in A-share/HK markets
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
            # Use a small epsilon for score_val to prevent log(0) issues in geometric mean
            # though since we divide by 100, score_val/100 is in [0, 1].
            # We use max(score_val, 0.0001) to ensure zero scores don't wipe out the whole product
            # unless they are intended to. However, per previous logic, a 0 score is a valid penalty.
            # To allow zero-scores to work in geometric mean, we scale them:
            normalized_val = max(score_val / 100.0, 0.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # If product is zero (due to a 0 score), result will be 0.
            # We use the exponent (1/total_weight) to normalize the weights back to a [0, 1] range.
            # Since product_of_powers is (S1^w1 * S2^w2...), the result is (product)^(1/sum_w).
            # This ensures that if all weights were 1, it's just the product.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use log-scale for valuation to handle outliers and non-linear relationship
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 30]
            pe_score = _log_score(pe, 30.0, 1.0)
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; usually indicates loss. We treat as low score but not zero.
            pe_score = 10.0 

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _log_score(pb, 5.0, 0.1)
        elif pb is not None and pb <= 0:
            pb_score = 5.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        roe_score = 0.0
        if roe is not None:
            # ROE can be negative, but for scoring we map it. 
            # Assume -20% is worst (0) and 30% is best (100).
            roe_score = _linear_score(roe * 100, 30.0, -20.0)
        
        de_score = 0.0
        if debt_equity is not None:
            # Lower debt is better. Assume 2.0 (200%) is worst, 0.0 is best.
            de_score = _linear_score(debt_equity, 0.0, 2.0)
        else:
            de_score = 50.0

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a growth-adjusted valuation metric
        peg_score = 0.0
        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, lower is better.
            peg_score = _linear_score(1/peg, 2.0, 0.1) # Invert because lower PEG is better
        elif peg is not None and peg <= 0:
            peg_score = 50.0

        roe_growth_score = 0.0
        if roe is not None:
            # High ROE often indicates efficient growth
            roe_growth_score = _linear_score(roe * 100, 25.0, -10.0)

        return (peg_score + roe_growth_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending (higher score is better rank/rank 1)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results