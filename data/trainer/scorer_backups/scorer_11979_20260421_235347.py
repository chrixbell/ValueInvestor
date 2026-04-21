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
    "quality": 0.25,   # Increased to emphasize fundamental strength
    "growth": 0.25,    # Increased to capture expansionary potential
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent 0.0 from zeroing out the whole product
            # but allow it to penalize heavily.
            normalized_val = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # In the geometric mean (S1^w1 * S2^w2...), if sum(weights) is not 1,
            # we need to handle the exponent correctly. For a normalized result:
            # If weights sum to 1, product_of_powers is the score.
            # To ensure scale consistency:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use a combination of PE and PB to capture valuation. 
        # We use the minimum (best) available ratio to avoid being penalized solely by one outlier.
        pe = result.valuation.pe_ratio if result.valuation.pe_ratio is not None and result.valuation.pe_ratio > 0 else None
        pb = result.valuation.pb_ratio if result.valuation.pb_ratio is not None and result.valuation.pb_ratio > 0 else None
        
        # Using fixed bounds for linear interpolation to prevent extreme outlier influence.
        # PE: 0-40, PB: 0-10.
        if pe is not None and pb is not None:
            # Combine into a single metric or pick the more conservative one.
            # Here we use PB as primary for assets and PE for earnings.
            return _linear_score(pb, 1.0, 8.0)
        elif pe is not None:
            return _linear_score(pe, 5.0, 30.0)
        elif pb is not None:
            return _linear_score(pb, 1.0, 8.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # Scale ROE (assume -20% to 40%) and Debt-to-Equity (assume 0 to 2.0)
        # Higher ROE is better, lower debt_to_equity is better.
        roe_s = _linear_score(roe * 100, 5.0, 30.0) # If ROE is 20% -> 20
        # For debt, we want lower to be higher score. We flip the logic:
        # If debt is 0, it's best (100). If debt is 2.0, it's worst (0).
        debt_s = _linear_score(1.0 / (debt_equity + 0.1), 1.0, 0.2) # This is tricky with linear_score
        # Let's use a simpler approach:
        debt_s = max(0.0, min(100.0, (2.0 - debt_equity) / 2.0 * 100.0)) if debt_equity > 0 else 100.0
        
        return (roe_s * 0.6) + (debt_s * 0.4)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        if peg is not None:
            peg_s = _linear_score(peg, 0.5, 3.0)
            # PEG is low = good. So we want to map 'peg' to a score where lower peg -> higher score.
            # _linear_score(value, best, worst) where best=0.5, worst=3.0
            # If peg=0.5 -> 100. If peg=3.0 -> 0.
            return (peg_s * 0.7) + (_linear_score(roe * 100, 5.0, 30.0) * 0.3)
        else:
            return _linear_score(roe * 100, 5.0, 30.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending (higher is better)
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results