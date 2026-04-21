"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Increased to capture expansion potential
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to avoid log(0) issues in geometric mean if score_val is 0
            # We treat a 0 score as extremely low but allow growth/quality through.
            normalized_score = max(score_val, 0.0) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers is already essentially the weighted geometric mean 
            # because we are multiplying (S/100)^w. To get it back to 100 scale:
            # (Product)^{1/total_weight} * 100. However, since sum(weights) is not necessarily 1,
            # we must handle the exponent. If weights sum to 1, product_of_powers is already scaled.
            # To be robust for any weight sum:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle cases where PE might be negative or None
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 25] for high score
            pe_score = _linear_score(pe, 1.0, 25.0)
        elif pe is not None and pe <= 0:
            # Negative PE (profitable) is often good, but hard to score linearly. 
            # We treat it as a high-value signal if we assume it's a turnaround.
            # For simplicity, map to a high value if PE is negative but exists.
            pe_score = 70.0 

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.5, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 50.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE score: higher is better
        roe_score = _linear_score(roe, 0.05, 0.30) # 5% to 30%
        
        # Debt score: lower is better. Handle potential negative debt/equity (negative equity)
        if debt_equity is not None and debt_equity > 0:
            debt_score = _linear_score(debt_equity, 0.1, 2.0) # 0.1 to 2.0
        else:
            debt_score = 50.0

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG score: lower is better (growth at reasonable price)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 3.0)
        else:
            peg_score = 50.0

        # ROE as a proxy for quality-growth
        roe_score = _linear_score(roe, 0.05, 0.25)

        return (peg_score + roe_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results