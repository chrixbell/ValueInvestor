"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to prioritize stable businesses
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent zero-out in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # If product is 0, result should be 0.
            if product_of_powers == 0:
                composite_score = 0.0
            else:
                # Scale back to [0, 100]
                composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use forward PE if available, else trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Default values for comparison if None
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Scoring: lower PE/PB is better
        # Target ranges: PE 5-30, PB 1-5
        pe_score = _linear_score(pe, best=5.0, worst=30.0)
        pb_score = _linear_score(pb, best=1.0, worst=5.0)
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE score: higher is better (Target 5% to 25%)
        roe_score = _linear_score(roe * 100, best=25.0, worst=5.0)
        # Debt score: lower is better (Target 0% to 100%)
        debt_score = _linear_score(debt_eq * 100, best=20.0, worst=100.0)

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0

        # ROE component for growth potential
        roe_score = _linear_score(roe * 100, best=20.0, worst=0.0)
        # PEG component: lower is better (Target 0.5 to 2.0)
        peg_score = _linear_score(peg, best=0.5, worst=2.0)

        return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results