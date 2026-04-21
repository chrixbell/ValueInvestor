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
    "quality": 0.25,   # Increased to prioritize fundamental strength
    "growth": 0.20,    # Retained growth component
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst:float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
    if value <= 0 or best <= 0 or worst <= 0:
        return 0.0

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0
        
        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Using an additive weighted average to prevent a single zero-score 
        # from wiping out the entire score (unlike geometric mean), 
        # while still allowing high quality to offset low value.
        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process)
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        sum_weighted_scores = 0.0
        for k, score_val in weighted_scores_to_process.items():
            sum_weighted_scores += score_val * self.weights[k]

        result.composite_score = sum_weighted_scores / total_weight
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use PE as primary, fallback to PB. 
        # We want low PE/PB to be high score.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Define reasonable bounds for Chinese A-share/HK markets
        # PE: 0 to 50 (low is good), PB: 0 to 15
        if pe is not None and pe > 0:
            # Standardize PE to [0, 100] where low is high score.
            # Using a soft threshold: PE of 1 -> 100, PE of 40 -> 0
            return _linear_score(pe, best=1.0, worst=40.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, best=0.5, worst=10.0)
        elif result.valuation.dividend_yield is not None:
            return _linear_score(result.valuation.dividend_yield, best=5.0, worst=0.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Leverage."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # Score components
        roe_score = 0.0
        debt_score = 0.0

        if roe is not None:
            # ROE typically ranges from -10% to 50%+
            roe_score = _linear_score(roe, best=0.25, worst=-0.10)
        
        if debt_equity is not None:
            # Lower debt-to-equity is better. 
            # Note: linear_score(value, best=0, worst=2) where value is debt.
            # We flip the logic: low debt = high score.
            debt_score = _linear_score(debt_equity, best=0.1, worst=1.5)
        elif result.financials.current_ratio is not None:
            debt_score = _linear_score(result.financials.current_ratio, best=2.0, worst=0.5)

        # Combine: Quality is heavily weighted towards ROE
        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        growth_score = 0.0
        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, lower is better for growth-value hybrid
            growth_score = _linear_score(peg, best=0.5, worst=3.0)
        elif roe is not None:
            # If no PEG, use ROE as a proxy for growth potential
            growth_score = _linear_score(roe, best=0.20, worst=-0.10)
        else:
            growth_score = 50.0

        return growth_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results