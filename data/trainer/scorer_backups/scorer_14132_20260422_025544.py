"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced value weight
    "quality": 0.25,   # Increased quality to capture stability
    "growth": 0.25,    # Balanced growth weight
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
            # We use a small epsilon to avoid math domain errors with 0.0 in geometric mean
            # but allow the score to be truly low if a factor is 0.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100].
            # Using (product_of_powers) directly since the weights are already applied in the power.
            # If we want (S1^w1 * S2^w2)^(1/sum_w), and weights are normalized to sum=1, 
            # then product_of_powers is already the result.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle potential None or negative values (for loss-making companies)
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Score based on low PE and low PB
        # We use a simple heuristic: lower is better. 
        # Typical healthy ranges for A-shares/HK: PE [5, 30], PB [1, 5]
        pe_score = _linear_score(pe, 5.0, 30.0)
        pb_score = _linear_score(pb, 1.0, 5.0)
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE: higher is better (e.g., 5% to 25%)
        roe_score = _linear_score(roe, 0.05, 0.25)
        # Debt to Equity: lower is better (e.g., 0.1 to 1.5)
        de_score = _linear_score(debt_equity, 0.1, 1.5)
        # Invert de_score because lower debt is better
        de_score = 100.0 - de_score
        
        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG: lower is better (e.g., 0.5 to 2.0)
        # Ensure peg is positive for linear scoring logic if it's negative (growth stocks)
        peg = max(0.1, peg)
        peg_score = _linear_score(peg, 0.5, 2.0)
        # Invert peg_score because lower PEG is better
        peg_score = 100.0 - peg_score
        
        # ROE as a proxy for internal growth/efficiency
        roe_score = _linear_score(roe, 0.05, 0.25)
        
        return (peg_score + roe_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results