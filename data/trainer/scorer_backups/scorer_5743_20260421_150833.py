"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly from 0.65 to balance with quality/growth
    "quality": 0.30,   # Increased weight to capture more stable returns
    "growth": 0.20,    # Maintained growth component
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent zero-score annihilation in geometric mean 
            # while still allowing low scores to weigh down the composite.
            normalized_score = max(0.01, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula for weighted factors: (Π S_i^w_i) ^ (1 / Σ w_i)
            # Since we used normalized scores [0, 1], the result is in [0, 1].
            # We scale it back to [0, 100].
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # We use a mix of forward PE and PB for valuation. 
        # Low PE/PB is better.
        pe = result.valuation.pe_forward or result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Default fallback values for comparison
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Score based on how low the multiples are
        # We want to reward lower PE/PB. 
        # Mapping: Pe=5 -> high score, Pe=50 -> low score.
        pe_score = _linear_score(1.0/pe, 1.0/50.0, 1.0/2.0) # Using inverse to handle 'lower is better'
        # Actually, let's use a more direct linear approach for simplicity in the interpolation logic:
        # We define a range where 5 is 'best' and 40 is 'worst'.
        pe_score = _linear_score(pe, 5.0, 40.0) # Note: _linear_score(val, best, worst) is (val-worst)/(best-worst)
        # Wait, the logic in _linear_score is: score = (value - worst) / (best - worst) * 100
        # If value=5, best=5, worst=40 -> (5-40)/(5-40) * 100 = 100. Correct.
        
        # Recalculating using the provided _linear_score logic:
        pe_s = _linear_score(pe, 5.0, 40.0)
        pb_s = _linear_score(pb, 0.5, 5.0)
        
        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe or 0.0
        debt_equity = result.financials.debt_to_equity or 0.5
        
        # ROE: Higher is better. Range [0% to 30%]
        roe_s = _linear_score(roe, 30.0, -10.0)
        # Debt: Lower is better. Range [0 to 2.0]
        debt_s = _linear_score(debt_equity, 0.0, 2.0)
        
        return (roe_s + debt_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe or 0.0
        
        # If PEG is None, we rely on ROE. If PEG exists, it's a primary growth metric.
        if peg is not None and peg > 0:
            # Low PEG is better. Range [0.5 to 3.0]
            peg_s = _linear_score(peg, 0.5, 3.0)
            # Use ROE as a secondary component
            roe_s = _linear_score(roe, 20.0, -10.0)
            return (peg_s + roe_s) / 2.0
        else:
            # Fallback to just ROE-based growth proxy
            return _linear_score(roe, 25.0, -5.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results