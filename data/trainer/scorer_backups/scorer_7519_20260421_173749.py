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
    "quality": 0.25,   # Increased to reward companies with better balance sheets
    "growth": 0.25,    # Increased to reward companies with higher earnings potential
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

        # To prevent a single zero score from wiping out the entire composite via geometric mean,
        # we use an epsilon-adjusted geometric mean approach.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use (score + epsilon) to ensure that a 0 score doesn't zero out the entire product
            # while still heavily penalizing it.
            normalized_val = (score_val / 100.0) + epsilon
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Calculate geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # Ensure we don't exceed 100 due to epsilon
            composite_score = min(100.0, composite_score)
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE might be negative or None
        pe_val = pe if (pe is not None and pe > 0) else 20.0
        pb_val = pb if (pb is not None and pb > 0) else 1.0

        # Scoring: Lower PE/PB is better
        # We use linear interpolation between a 'good' range and a 'bad' range.
        pe_score = _linear_score(pe_val, 10.0, 40.0)
        pb_score = _linear_score(pb_val, 1.0, 6.0)
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better (range 0% to 30%)
        roe_score = _linear_score(roe * 100, 25.0, 0.0) # Note: _linear_score handles best/worst
        # But wait, the existing _linear_score uses (val - worst)/(best-worst). 
        # For ROE, higher is best. Let's re-map:
        roe_score = _linear_score(roe * 100, 30.0, -5.0) # 30% -> 100, -5% -> 0
        
        # Debt/Equity: Lower is better (range 0 to 1.5)
        de_score = _linear_score(debt_equity, 0.2, 1.5)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG ratio and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.1

        # PEG: Lower is better (range 0.5 to 3.0)
        peg_score = _linear_score(peg, 0.5, 3.0)
        
        # ROE component for growth (higher is better)
        roe_score = _linear_score(roe * 100, 20.0, -5.0)

        return (peg_score + roe_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results