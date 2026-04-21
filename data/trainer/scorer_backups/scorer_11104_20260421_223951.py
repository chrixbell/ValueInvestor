"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced value weight to allow for quality/growth expansion
    "quality": 0.25,   # Increased quality weight to favor robust balance sheets
    "growth": 0.25,    # Balanced with quality to capture sustainable expansion
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k] 
            total_weight += weight
            # We add a small epsilon to avoid zero-out issues in geometric mean 
            # while still allowing the score to drop significantly.
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores [0,1] scaled back to 100
            # Note: The previous implementation had a (1/total_weight) exponent 
            # which is mathematically correct for the product of powers.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or negative values (e.g., negative PE)
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Map PE to a score where lower is better (range 0-30)
            pe_score = _linear_score(pe, best=5.0, worst=30.0)
        
        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Map PB to a score where lower is better (range 0-30)
            pb_score = _linear_score(pb, best=1.5, worst=6.0)
            
        # If one is missing, use the available one; otherwise average
        if pe is None: return pb_score
        if pb is None: return pe_score
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better
        roe_score = _linear_score(roe, best=0.20, worst=0.05)
        # Debt-to-Equity: Lower is better (clamped to avoid negative issues)
        de_score = _linear_score(debt_eq, best=0.2, worst=1.5)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.3

        # PEG: Lower is better (indicates growth is cheap)
        peg_score = _linear_score(peg, best=0.5, worst=2.5)
        # Margin: Higher is better
        margin_score = _linear_score(margin, best=0.4, worst=0.1)

        return (peg_score + margin_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results