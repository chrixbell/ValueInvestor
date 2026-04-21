"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth stability
    "quality": 0.25,   # Increased to prioritize companies with better balance sheets
    "growth": 0.25,    # Balanced weight for growth potential
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
            # Using a small epsilon to prevent 0 score from zeroing out the whole product
            # while still allowing bad scores to drag down the geometric mean.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of (S/100)^w is equal to the geometric mean if the sum of weights is 1.
            # We scale by 100 to return to [0, 100].
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle negative PE (loss making) by treating it as very poor value
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Using a wider range for PE to capture value stocks better
            pe_score = _linear_score(pe, 5.0, 30.0)
        elif pe is not None and pe <= 0:
            pe_score = 10.0 # Small score for loss makers, but not zero to allow growth/quality to carry

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1.0, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 50.0

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        roe_score = 0.0
        if roe is not None:
            # ROE can be negative, but we map positive ROE to the score range.
            roe_score = _linear_score(roe, 0.05, 0.25)
        
        de_score = 0.0
        if debt_equity is not None:
            # Lower debt to equity is better.
            de_score = _linear_score(debt_equity, 0.1, 1.5)
            # Invert: high de_score (low debt) should be better. 
            # Actually, _linear_score with best=0.1 and worst=1.5 means 
            # low debt gets high score. Correct.

        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Revenue Growth."""
        peg = result.valuation.peg_ratio
        # We don't have explicit revenue growth, but we can use ROE as a proxy for internal growth
        # or assume PEG is the primary driver. 
        
        peg_score = 0.0
        if peg is not None and peg > 0:
            # Ideally PEG < 1. We map PEG of 0.5 to 100 and 3.0 to 0.
            peg_score = _linear_score(1/peg, 0.5, 3.0) # Using inverse to reward lower PEG
            # Wait, _linear_score(value, best, worst) where best is 0.5 and worst is 3.
            # If value is 0.5, score=100. If 3.0, score=0.
            # But if value is 0.1, it would be > 100 (clamped to 100).
        elif peg is not None and peg <= 0:
             peg_score = 20.0

        return peg_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results