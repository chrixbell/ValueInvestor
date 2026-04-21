"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Slightly reduced to allow for more balance
    "quality": 0.25,   # Increased quality weight to prioritize robust balance sheets
    "growth": 0.30,    # Growth remains important for forward returns
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

        # Use a small epsilon to prevent zero-product issues in geometric mean 
        # while still penalizing very low scores.
        epsilon = 0.01
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] range. We use max(epsilon, ...) to ensure
            # that a zero in one category doesn't completely wipe out the score, 
            # but still heavily penalizes it.
            normalized_score = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The formula (product^(1/total_weight)) * 100 simplifies to:
            # If weights sum to 1, it's just product * 100.
            # We calculate the geometric mean of the normalized scores first.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Default fallback values if data is missing
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Scoring: Lower PE/PB is better
        # We use a simple linear scale for comparison
        pe_score = _linear_score(20.0, 5.0, 40.0) # Lower PE is higher score
        pb_score = _linear_score(pb, 1.0, 5.0)   # Lower PB is higher score
        
        # Combine: We want to favor lower PE but also check if PB is reasonable.
        # If PE is very low, it's great. 
        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        current_ratio = result.financials.current_ratio if result.financials.current_ratio is not None else 1.0

        # ROE Score: Higher is better (using linear mapping)
        roe_score = _linear_score(0.15, 0.05, 0.25) # 15% is mid-point
        
        # Leverage Score: Lower debt/equity is better
        leverage_score = _linear_score(0.5, 0.2, 1.0)
        
        # Liquidity Score: Current ratio should be > 1.0
        liquidity_score = _linear_score(current_ratio, 0.5, 2.0)

        return (roe_score * 0.5) + (leverage_score * 0.3) + (liquidity_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG: Lower is better (but avoid negative/zero handling via logic)
        # If PEG is 1.0, it's neutral (50). If < 1, good. If > 1, bad.
        peg_score = _linear_score(1.0, 0.5, 3.0)
        
        # ROE as a proxy for growth/efficiency
        roe_score = _linear_score(0.15, 0.05, 0.25)

        return (peg_score * 0.6) + (roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results