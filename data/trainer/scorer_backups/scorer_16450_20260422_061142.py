"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Adjusted to balance with quality/growth
    "quality": 0.30,   # Increased weight for stability
    "growth": 0.20,    # Maintained growth component
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

        # We use an additive weighted average for the composite score to prevent 
        # a single zero-score in one category from nuking the entire composite score, 
        # while still allowing high-quality stocks to shine.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PEG and higher dividend yield are better."""
        # Use multiple valuation metrics to avoid over-reliance on one (e.g., negative PE)
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio
        div = result.valuation.dividend_yield

        scores = []
        # PE: Lower is better (clamp to avoid extreme negatives)
        if pe is not None and pe > 0:
            scores.append(_linear_score(1/pe, 1/5, 1/50)) # Target PE range 5-50
        elif pe is not None and pe <= 0:
            scores.append(100.0 if pe < -10 else 50.0)
        
        if pb is not None and pb > 0:
            scores.append(_linear_score(1/pb, 1/1, 1/10))
            
        if peg is not None and peg > 0:
            scores.append(_linear_score(1/peg, 1/1, 1/3))

        if div is not None:
            scores.append(_linear_score(div, 0.05, 0.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margins, lower leverage."""
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity
        
        scores = []
        if roe is not None:
            # Map ROE to score (e.g., 20% is great, -10% is bad)
            scores.append(_linear_score(roe, 0.20, -0.10))
        if margin is not None:
            scores.append(_linear_score(margin, 0.15, -0.05))
        if debt is not None:
            # Lower debt to equity is better. 0 is best, 2.0 is worst.
            scores.append(_linear_score(1/(debt + 0.1 if debt > 0 else 0.1), 1/2.5, 1/0.5))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and growth potential."""
        # Growth is often captured by the interaction of profitability and investment.
        # Here we use ROE as a proxy for efficient growth-reinvestment.
        roe = result.financials.roe
        rev_growth = None # Not explicitly in data, but we can use ROE/Margin as proxies
        
        scores = []
        if roe is not None:
            # Growth-oriented stocks often have higher ROE expectations
            scores.append(_linear_score(roe, 0.15, -0.05))
        
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results