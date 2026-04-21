"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.55,     # Slightly reduced value weight to allow quality/growth more room
    "quality": 0.25,   # Increased quality to capture stable earners
    "growth": 0.20,    # Maintaining growth weight
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
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

        # We use a weighted arithmetic mean for the composite score to prevent 
        # a single zero-score (e.g. in one sub-factor) from wiping out the entire 
        # score, which can happen with geometric means in noisy financial data.
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
        """Lower PE/PB/PS is better."""
        v = result.valuation
        f = result.financials
        
        # Priority 1: PE Ratio (if positive)
        if v.pe_ratio is not None and v.pe_ratio > 0:
            return _linear_score(v.pe_ratio, 5, 30)
        
        # Priority 2: PB Ratio
        if v.pb_ratio is not None and v.pb_ratio > 0:
            return _linear_score(v.pb_ratio, 1, 5)
        
        # Priority 3: PS Ratio
        if v.ps_ratio is not None and v.ps_ratio > 0:
            return _linear_score(v.ps_ratio, 1, 5)

        # Fallback: Dividend Yield (Higher is better)
        if v.dividend_yield is not None:
            return _linear_score(v.dividend_yield, 5, 0) # Note: linear handles direction via best/worst logic
            # Actually _linear_score uses (val-worst)/(best-worst). 
            # If best=5, worst=0, and val=2: (2-0)/(5-0)*100 = 40. Correct.

        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        f = result.financials
        v = result.valuation

        # ROE is a primary quality metric
        if f.roe is not None:
            return _linear_score(f.roe, 15, -5)
        
        # Debt to Equity
        if f.debt_to_equity is not None:
            return _linear_score(f.debt_to_equity, 0.5, 2.0)

        # Gross Margin
        if f.gross_margin is not None:
            return _linear_score(f.gross_margin, 40, 0)

        return 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth or favorable PEG."""
        v = result.valuation
        f = result.financials

        # Check PEG (Growth/PE) - Low is better
        if v.peg_ratio is not None and v.peg_ratio is not None and v.peg_ratio > 0:
            return _linear_score(v.peg_ratio, 0.5, 2.0)

        # Use ROE as a proxy for growth if PEG is unavailable
        if f.roe is not None:
            return _linear_score(f.roe, 20, 0)

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