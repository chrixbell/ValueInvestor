"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced from 0.65 to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture more stability
    "growth": 0.30,    # Increased to capture higher potential returns
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

        # We use a weighted arithmetic mean for the composite score to prevent 
        # a single zero-score in one category from zeroing out the entire profile,
        # while still allowing weights to dictate importance.
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
        """Calculates value score using PE and PB ratios."""
        # We prioritize PE for growth-oriented companies but use PB as a stabilizer.
        # If PE is missing, we fall back to PB or PS.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Best/Worst thresholds for linear/log scoring
        # Note: For PE/PB, lower is better.
        if pe is not None and pe > 0:
            # Use a soft-clamped range for PE (e.g., 1 to 40)
            # We use a simple linear comparison for variety in this iteration.
            v_score = _linear_score(pe, 1.0, 40.0)
            # Invert because lower PE is better
            return 100.0 - v_score
        elif pb is not None and pb > 0:
            v_score = _linear_score(pb, 0.5, 15.0)
            return 100.0 - v_score
        elif ps is not None and ps > 0:
            v_score = _linear_score(ps, 0.5, 10.0)
            return 100.0 - v_score
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE is a primary driver of quality
        roe_score = 0.0
        if roe is not None:
            # Map ROE (assume -20% to 40% range) to [0, 100]
            roe_score = _linear_score(roe, 30.0, -10.0)
        
        # Debt/Equity: lower is better
        de_score = 0.0
        if debt_equity is not None:
            # Map DE (assume 0 to 2.0 range) to [0, 100]
            de_score = _linear_score(debt_equity, 0.2, 1.5)
            de_score = 100.0 - de_score # Invert

        # Combine: ROE is much more important for quality
        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using PEG and ROE interaction."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a classic growth/value hybrid
        peg_score = 50.0
        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, < 1.0 is good, > 2.0 is bad
            peg_score = _linear_score(peg, 0.5, 2.5)
            peg_score = 100.0 - peg_score

        # Use ROE as a proxy for internal growth capability
        roe_growth = 0.0
        if roe is not None:
            roe_growth = _linear_score(roe, 20.0, -5.0)

        return (peg_score * 0.6) + (roe_growth * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results