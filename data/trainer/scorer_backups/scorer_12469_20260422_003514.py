"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced slightly to give more room to quality/growth
    "quality": 0.25,   # Increased to emphasize fundamental stability
    "growth": 0.30,    # Increased to capture expansion potential
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

        # Using an additive weighted average for stability in the presence of zeros
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB."""
        v = result.valuation
        if not v:
            return 50.0
        
        # Use PE as primary, PB as secondary if PE is unavailable or extreme
        if v.pe_ratio is not None and v.pe_ratio > 0:
            # Low PE is good
            pe_score = _linear_score(v.pe_ratio, 5.0, 40.0)
            if v.pb_ratio is not None and v.pb_ratio > 0:
                pb_score = _linear_score(v.pb_ratio, 1.0, 10.0)
                return (pe_score * 0.7 + pb_score * 0.3)
            return pe_score
        elif v.pb_ratio is not None and v.pb_ratio > 0:
            return _linear_score(v.pb_ratio, 1.0, 10.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        f = result.financials
        if not f:
            return 50.0

        # ROE is a primary quality metric
        roe = f.roe if f.roe is not None else 0.0
        # Debt/Equity: lower is better
        de = f.debt_to_equity if f.debt_to_equity is not None else 0.5

        # Normalize ROE (assume 20% is great, -10% is bad)
        roe_score = _linear_score(roe, -0.1, 0.25)
        # Normalize DE (assume 0.5 is great, 2.0 is bad)
        de_score = _linear_score(de, 0.0, 2.0)

        return (roe_score * 0.7 + de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        # PEG ratio (Growth/PE) - lower is often better for value-growth hybrid
        peg = v.peg_ratio if (v.peg_ratio is not None and v.peg_ratio > 0) else None
        roe = f.roe if f.roe is not None else 0.0

        if peg is not None:
            # PEG of 1.0 is neutral, < 1 is good, > 2 is bad
            peg_score = _linear_score(peg, 0.5, 2.5)
            roe_score = _linear_score(roe, -0.1, 0.3)
            return (peg_score * 0.6 + roe_score * 0.4)
        else:
            # Fallback to ROE-based growth score if PEG is missing
            return _linear_score(roe, -0.1, 0.3)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results