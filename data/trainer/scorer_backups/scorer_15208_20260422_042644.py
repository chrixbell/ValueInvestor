"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Adjusted to allow more weight for quality/growth
    "quality": 0.25,   # Increased to reward stability
    "growth": 0.25,    # Balanced with quality
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

        # Momentum is a placeholder
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

        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        # We use an additive weighted average for the base score to prevent 
        # a single zero-score factor from completely wiping out the result,
        # but we apply a penalty for low scores to maintain some non-linearity.
        total_weighted_sum = 0.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weighted_sum += (score_val * weight)

        # The composite score is the weighted average
        composite_score = total_weighted_sum / total_weight
        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield are better."""
        v = result.valuation
        f = result.financials
        
        # Primary value drivers
        pe_score = 0.0
        if v.pe_ratio is not None and v.pe_ratio > 0:
            # Using a reasonable range for PE (1 to 40)
            pe_score = _linear_score(v.pe_ratio, 5.0, 30.0)
            pe_score = 100.0 - pe_score # Lower is better
        elif v.pe_ratio is not None and v.pe_ratio <= 0:
            pe_score = 100.0 # Negative PE is often a value signal (though risky)
        else:
            pe_score = 50.0

        pb_score = 0.0
        if v.pb_ratio is not None:
            pb_score = _linear_score(v.pb_ratio, 1.0, 5.0)
            pb_score = 100.0 - pb_score
        else:
            pb_score = 50.0

        div_score = 0.0
        if v.dividend_yield is not None:
            div_score = _linear_score(v.dividend_yield, 2.0, 8.0)
        else:
            div_score = 50.0

        # Combine factors: PE and PB are weighted equally, dividend is a bonus
        return (pe_score * 0.4 + pb_score * 0.4 + div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin and lower leverage are better."""
        f = result.financials
        v = result.valuation

        roe_score = 0.0
        if f.roe is not None:
            roe_score = _linear_score(f.roe, 5.0, 25.0)
        else:
            roe_score = 50.0

        margin_score = 0.0
        if f.net_margin is not None:
            margin_score = _linear_score(f.net_margin, 5.0, 20.0)
        else:
            margin_score = 50.0

        debt_score = 100.0
        if f.debt_to_equity is not None:
            # Lower debt to equity is better
            debt_score = _linear_score(f.debt_to_equity, 0.0, 1.5)
            debt_score = 100.0 - debt_score
        else:
            debt_score = 50.0

        return (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG ratio."""
        v = result.valuation
        f = result.financials

        # Growth is often captured by ROE in steady-state, but let's use PEG as a proxy
        # for growth-at-a-reasonable-price.
        peg_score = 0.0
        if v.peg_ratio is not None and v.peg_ratio > 0:
            # PEG of 1.0 is ideal. Lower is better for growth-value combo.
            peg_score = _linear_score(v.peg_ratio, 0.5, 2.5)
            peg_score = 100.0 - peg_score
        else:
            peg_score = 50.0

        # Use ROE as a proxy for internal growth capability
        roe_growth = 0.0
        if f.roe is not None:
            roe_growth = _linear_score(f.roe, 5.0, 30.0)
        else:
            roe_growth = 50.0

        return (peg_score * 0.6 + roe_growth * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results