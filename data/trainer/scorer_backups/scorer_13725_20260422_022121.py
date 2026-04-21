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
    "quality": 0.25,   # Increased to prioritize stable businesses
    "growth": 0.25,    # Balanced with quality
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

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0,
        }

        # We use a weighted sum of normalized scores for stability in the ranking.
        # Using geometric mean with 0-scores can lead to extreme sensitivity.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_sum += score_val * weight
                total_weight += weight

        if total_weight > 0:
            # Normalize back to [0, 100]
            result.composite_score = weighted_sum / (total_weight / 100.0) if total_weight != 0 else 50.0
            # Ensure it stays in bounds if weights are weirdly scaled
            result.composite_score = max(0.0, min(100.0, result.composite_score))
        else:
            result.composite_score = 50.0

        return result

    def _value_score(self, result: ScreeningResult) -> float:
        v = result.valuation
        # Primary value drivers: PE and PB. 
        # We use a combination of PE and PB to capture different valuation aspects.
        pe = v.pe_ratio if (v.pe_ratio is not None and v.pe_ratio > 0) else None
        pb = v.pb_ratio if (v.pb_ratio is not None and v.pb_ratio > 0) else None
        
        # Fallback to neutral if no data
        if pe is None and pb is None:
            return 50.0
        
        # If we have PE, it's often a more direct driver for 6-month returns than PB.
        # We use reasonable bounds: PE of 5 to 30, PB of 0.5 to 10.
        if pe is not None:
            # Lower PE is better for value. 
            # We want a score where low PE -> high score.
            score = _linear_score(pe, best=5.0, worst=30.0)
            # If PE is very high (e.g., 100), score will be 0 via clamping.
            return score if pe <= 30.0 else 0.0
        else:
            return _linear_score(pb, best=0.5, worst=10.0)

    def _quality_score(self, result: ScreeningResult) -> float:
        f = result.financials
        # Quality is driven by ROE and Debt/Equity.
        roe = f.roe if (f.roe is not None) else 0.0
        debt_eq = f.debt_to_equity if (f.debt_to_equity is not None and f.debt_to_equity > 0) else 0.5
        
        # ROE Score: High is better. (Assume reasonable range -10% to 40%)
        # We map ROE linearly.
        roe_score = _linear_score(roe * 100, best=30.0, worst=-5.0)
        
        # Debt Score: Low is better. (Assume 0 to 100% range)
        debt_score = _linear_score(debt_eq * 100, best=20.0, worst=100.0)
        
        # Combine: 70% ROE, 30% Debt management
        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        v = result.valuation
        f = result.financials
        
        # Growth is often captured by PEG or Net Margin expansion/levels.
        peg = v.peg_ratio if (v.peg_ratio is not None and v.peg_ratio > 0) else None
        margin = f.net_margin if (f.net_margin is not None) else 0.0
        
        if peg is not None:
            # Lower PEG is better (growth relative to value)
            # Map PEG 0.5 -> 100, PEG 5.0 -> 0
            score = _linear_score(peg, best=0.5, worst=5.0)
            return score
        else:
            # If no PEG, use margin as a proxy for profitability/growth-readiness
            return _linear_score(margin * 100, best=20.0, worst=-5.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        # Sort descending (higher score = better rank)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results