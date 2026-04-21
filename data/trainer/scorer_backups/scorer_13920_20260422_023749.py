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
    "quality": 0.25,   # Increased to prioritize stable earners
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
    Values beyond the endpoints are clamped.
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

        # We use a weighted arithmetic mean for the composite score to ensure 
        # that a single zero-score (e.g., in one factor) doesn't destroy the 
        # entire score, while still allowing it to drag the average down.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # We add a tiny epsilon to score_val if it's 0 to prevent issues in potential 
            # geometric calculations, though we use arithmetic mean here for stability.
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use PEG as a tie-breaker/enhancer if available
        peg = result.valuation.peg_ratio
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Define reasonable bounds for Chinese markets
        # PE: 0 to 40 is a standard range; PB: 0 to 10
        # Using linear interpolation for simplicity in this iteration
        
        v_score = 50.0
        if pe is not None and pb is not None:
            # Combine PE and PB into a single valuation metric (simplified)
            # In reality, we want low PE and low PB. 
            # We'll treat them as separate components of the value score.
            pe_score = _linear_score(pe, 40.0, 0.1) if pe > 0 else 0.0
            pb_score = _linear_score(pb, 10.0, 0.1) if pb > 0 else 0.0
            v_score = (pe_score + pb_score) / 2.0
        elif pe is not None:
            v_score = _linear_score(pe, 40.0, 0.1) if pe > 0 else 0.0
        elif pb is not None:
            v_score = _linear_score(pb, 10.0, 0.1) if pb > 0 else 0.0
        elif peg is not None:
            v_score = _linear_score(peg, 3.0, 0.1) if peg > 0 else 0.0
            
        return v_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity

        q_score = 50.0
        # ROE is a primary quality indicator
        roe_val = roe if roe is not None else 0.0
        # Debt to Equity: lower is better. Map 2.0 (high) to 0 and 0.1 (low) to 100
        de_val = debt_eq if debt_eq is not None else 2.0
        
        # Normalize ROE (assuming 0-40% range) and DE
        roe_s = _linear_score(roe_val, 30.0, -10.0)
        de_s = _linear_score(de_val, 2.0, 0.0)
        
        # Balance ROE and Debt
        q_score = (roe_s * 0.7) + (de_s * 0.3)
        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        # Growth is often captured by high ROE or expanding margins
        roe = result.financials.roe
        margin = result.financials.net_margin
        peg = result.valuation.peg_ratio

        g_score = 50.0
        if roe is not None and peg is not None:
            # High ROE + Low PEG = classic growth value
            roe_s = _linear_score(roe, 25.0, -5.0)
            peg_s = _linear_score(peg, 2.0, 0.1) if peg is not None and peg > 0 else 0.0
            g_score = (roe_s * 0.5) + (peg_s * 0.5)
        elif roe is not None:
            g_score = _linear_score(roe, 25.0, -5.0)
        elif margin is not None:
            g_score = _linear_score(margin, 20.0, -5.0)
            
        return g_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results