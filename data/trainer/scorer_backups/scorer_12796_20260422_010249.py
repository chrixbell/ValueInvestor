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
    "quality": 0.25,   # Increased to capture fundamental strength
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

        # Using arithmetic mean of normalized scores to prevent a single zero 
        # (from one bad factor) from destroying the entire score, which can 
        # happen with geometric means in noisy financial data.
        total_weighted_val = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_val += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_val / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better. Uses log scale to handle wide ranges."""
        v = result.valuation
        # We prioritize PE and PB as primary value drivers
        if v.pe_ratio and v.pe_ratio > 0:
            return _log_score(1/v.pe_ratio, 1/0.1, 1/100) # Higher inverse PE is better
        if v.pb_ratio and v.pb_ratio > 0:
            return _log_score(1/v.pb_ratio, 1/0.1, 1/10)
        if v.ps_ratio and v.ps_ratio > 0:
            return _log_score(1/v.ps_ratio, 1/0.1, 1/10)
        
        # Fallback to dividend yield if valuation ratios are missing
        if v.dividend_yield and v.dividend_yield > 0:
            return _linear_score(v.dividend_yield, 10.0, 0.0)
            
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        f = result.financials
        score_components = []
        
        # ROE is a primary quality metric
        if f.roe is not None:
            score_components.append(min(100.0, max(0.0, (f.roe + 20) * 2))) # Assume -10% is 0, 30% is 100
        else:
            score_components.append(50.0)

        # Debt to Equity (lower is better)
        if f.debt_to_equity is not None:
            # 0 debt = 100, 2.0 debt = 0
            score_components.append(max(0.0, min(100.0, 100.0 - (f.debt_to_equity * 50.0))))
        else:
            score_components.append(50.0)

        # Net Margin
        if f.net_margin is not None:
            score_components.append(min(100.0, max(0.0, (f.net_margin + 10) * 5)))
        else:
            score_components.append(50.0)

        return sum(score_components) / len(score_components)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth potential (PEG/ROE interaction)."""
        v = result.valuation
        f = result.financials
        
        score = 50.0
        if v.peg_ratio and v.peg_ratio > 0:
            # Low PEG is very good (e.g., 0.5)
            score = _log_score(1/v.peg_ratio, 1/0.5, 1/5)
        elif f.roe is not None:
            # If no PEG, use ROE as a proxy for growth/efficiency
            score = min(100.0, max(0.0, (f.roe + 10) * 2))
            
        return score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results