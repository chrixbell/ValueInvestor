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
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Growth remains a key component
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
            # Use a small epsilon to prevent issues with zero scores in geometric mean
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Using PE as primary driver, fallback to PB or dividend yield if needed
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Handle negative PE (often indicates loss)
        if pe is not None and pe > 0:
            # High PE = low score, Low PE = high score. Target range [1, 50]
            pe_score = _linear_score(pe, best=1.0, worst=50.0)
            if pb is not None and pb > 0:
                pb_score = _linear_score(pb, best=1.0, worst=10.0)
                return (pe_score * 0.7) + (pb_score * 0.3)
            return pe_score
        
        if pb is not None and pb > 0:
            return _linear_score(pb, best=1.0, worst=10.0)
        
        if dy is not None and dy > 0:
            return _linear_score(dy, best=5.0, worst=0.0)
            
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and leverage."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        current_ratio = result.financials.current_ratio

        scores = []
        if roe is not None:
            # Higher ROE is better. Range [0, 40]%
            scores.append(_linear_score(roe, best=30.0, worst=-10.0))
        
        if debt_to_equity is not None and debt_to_equity > 0:
            # Lower debt-to-equity is better. Range [0, 200]%
            scores.append(_linear_score(debt_to_equity, best=0.0, worst=2.0))
            
        if current_ratio is not None and current_ratio > 0:
            # Higher current ratio is better. Range [0, 3]
            scores.append(_linear_score(current_ratio, best=2.0, worst=0.5))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using PEG and margin."""
        peg = result.valuation.peg_ratio
        margin = result.financials.net_margin

        scores = []
        if peg is not None and peg > 0:
            # Lower PEG is better. Range [0, 2]
            scores.append(_linear_score(peg, best=0.5, worst=3.0))
        
        if margin is not None:
            # Higher net margin is better. Range [-5%, 30%]
            scores.append(_linear_score(margin, best=20.0, worst=-5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results