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
    "quality": 0.25,   # Increased to prioritize robust balance sheets
    "growth": 0.25,    # Balanced with quality for a more holistic view
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

        # We use an arithmetic mean of normalized scores for more stability in ranking.
        # A zero score (e.g., in value) shouldn't necessarily kill the entire composite 
        # if quality and growth are stellar, but we still weight them according to the user.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Add a small epsilon to avoid issues with zero-valued factors in some math operations
            # though arithmetic mean handles 0 naturally.
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Using a mix of PE and PB for valuation. 
        # We use the best/worst bounds based on reasonable market ranges.
        pe = result.valuation.pe_ratio if result.valuation.pe_ratio is not None and result.valuation.pe_ratio > 0 else None
        pb = result.valuation.pb_ratio if result.valuation.pb_ratio is not None and result.valuation.pb_ratio > 0 else None
        ps = result.valuation.ps_ratio if result.valuation.ps_ratio is not None and result.valuation.ps_ratio > 0 else None

        # Score components
        scores = []
        if pe: scores.append(_linear_score(pe, 5.0, 40.0))
        if pb: scores.append(_linear_score(pb, 1.0, 8.0))
        if ps: scores.append(_linear_score(ps, 1.0, 5.0))
        
        # If no data, return neutral
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        
        # ROE Score (Targeting 15-25%)
        roe_s = _linear_score(roe, 15.0, 30.0)
        # Debt Score (Lower is better: target < 1.0)
        debt_s = _linear_score(debt_equity, 0.5, 2.0)
        # Margin Score (Higher is better: target 10-20%)
        margin_s = _linear_score(margin, 5.0, 20.0)
        
        return (roe_s * 0.5) + (debt_s * 0.3) + (margin_s * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # Growth-oriented: If PEG is available, it's a strong signal
        if peg is not None:
            # PEG: lower is better (1.0 is neutral, < 1 is good)
            # We use a slightly higher 'worst' to avoid extreme sensitivity
            peg_s = _linear_score(peg, 0.5, 2.5)
            # Combine with ROE to ensure we aren't just buying "cheap" low-growth stocks
            roe_s = _linear_score(roe, 5.0, 25.0)
            return (peg_s * 0.6) + (roe_s * 0.4)
        else:
            # Fallback to ROE-based growth proxy
            return _linear_score(roe, 5.0, 30.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results