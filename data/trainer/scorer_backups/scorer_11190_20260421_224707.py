"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more focus on fundamentals
    "quality": 0.25,   # Increased quality weight to filter for better-managed firms
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            norm_score = max(0.0, score_val) / 100.0
            product_of_powers *= (norm_score ** weight)

        if total_weight > 0:
            # If product is 0, the result should be 0. Otherwise calculate via power rule.
            if product_of_powers > 0:
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            else:
                composite_score = 0.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        # Focus on combination of PE and PB to capture different valuation styles
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use a fallback for negative PE (common in China) - assign low but not zero score
        # We'll treat negative PE as very high (bad) for valuation scoring.
        pe_val = pe if pe is not None and pe > 0 else 999.0
        pb_val = pb if pb is not None and pb > 0 else 999.0
        
        # Score components: lower PE/PB is better. 
        # Using reasonable bounds for A-share/HK markets.
        score_pe = _linear_score(pe_val, best=5.0, worst=40.0)
        score_pb = _linear_score(pb_val, best=1.0, worst=10.0)
        
        # If values are extreme (e.g., PE=999), linear_score returns 0.
        return (score_pe * 0.6) + (score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 100.0
        
        # ROE is a primary quality metric. High ROE is good.
        # Debt-to-equity: lower is better.
        score_roe = _linear_score(roe, best=20.0, worst=-5.0)
        score_margin = _linear_score(margin, best=15.0, worst=-5.0)
        score_debt = _linear_score(debt, best=0.5, worst=3.0)
        
        return (score_roe * 0.4) + (score_margin * 0.3) + (score_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        # Growth is often captured by PEG or revenue growth. 
        # Since we don't have explicit growth rates, we use PEG and ROE as proxies.
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 10.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # Lower PEG = better growth value. Higher ROE = higher quality growth potential.
        score_peg = _linear_score(peg, best=0.5, worst=3.0)
        score_roe = _linear_score(roe, best=15.0, worst=0.0)
        
        return (score_peg * 0.6) + (score_roe * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results