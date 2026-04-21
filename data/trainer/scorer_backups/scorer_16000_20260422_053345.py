"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow room for quality/growth
    "quality": 0.25,   # Increased to capture stable returns
    "growth": 0.25,    # Increased to capture expansion potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Values beyond the endpoints are clamped.
    """
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

    def score(self, result: Screening_Result) -> Screening_Result:
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

        # We use an arithmetic mean for the final composite to prevent 
        # a single zero-score (from one bad metric) from wiping out the entire score.
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
        """Lower PE/PB and higher dividend yield are better."""
        v = result.valuation
        f = result.financials
        
        # Use PE as primary, fallback to PB or PS
        if v.pe_ratio and v.pe_ratio > 0:
            # Use PEG as a modifier if available (lower is better)
            if v.peg_ratio and v.peg_ratio > 0:
                # High PE is penalized more if growth (PEG) isn't there
                return _linear_score(1.0/v.pe_ratio, 1.0/5.0, 1.0/40.0)
            return _linear_score(1.0/v.pe_ratio, 1.0/5.0, 1.0/40.0)
        elif v.pb_ratio and v.pb_ratio > 0:
            return _linear_score(1.0/v.pb_ratio, 1.0/1.0, 1.0/10.0)
        elif v.ps_ratio and v.ps_ratio > 0:
            return _linear_score(1.0/v.ps_ratio, 1.0/1.0, 1.0/20.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and margins, lower debt."""
        f = result.financials
        q_score = 0.0
        count = 0

        if f.roe:
            q_score += _linear_score(f.roe, 5.0, 30.0)
            count += 1
        if f.net_margin:
            q_score += _linear_score(f.net_margin, 2.0, 20.0)
            count += 1
        if f.debt_to_equity:
            # For debt, lower is better. We invert the logic for linear_score usage.
            # If debt is 0, it's 'best'. 
            q_score += _linear_score(1.0 / (f.debt_to_equity + 0.1), 1.0/1.0, 1.0/5.0)
            count += 1

        return q_score / count if count > 0 else 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth metrics."""
        f = result.financials
        v = result.valuation
        g_score = 0.0
        count = 0

        # Growth is often captured in PEG or ROE expansion, but we use ROE/Revenue here
        if f.roe:
            g_score += _linear_score(f.roe, 5.0, 25.0)
            count += 1
        if v.pe_forward and v.pe_forward > 0:
            # Forward PE discount as a proxy for growth expectation
            g_score += _linear_score(1.0/v.pe_forward, 1.0/30.0, 1.0/5.0)
            count += 1

        return g_score / count if count > 0 else 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results