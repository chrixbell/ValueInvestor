"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to reward robust balance sheets
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
            # Using (score + 1e-6) to prevent math errors with zero scores in geometric mean
            # while keeping the 0-100 scaling logic.
            normalized_score = (max(0.0, score_val) / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # If the product is 0, we handle it to avoid issues with power of zero.
            if product_of_powers <= 0:
                composite_score = 0.0
            else:
                # The geometric mean formula for weighted values is (PI x_i^w_i)^(1 / SUM w_i)
                # Since we normalized by 100, we multiply by 100 at the end.
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback for negative/None PE (common in loss-making companies)
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Map PE to score (lower is better)
            pe_score = _linear_score(pe, 5.0, 40.0)
            
        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Map PB to score (lower is better)
            pb_score = _linear_score(pb, 1.0, 5.0)
            
        return (pe_score * 0.6 + pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # Higher ROE is better
        roe_score = _linear_score(roe * 100, 5.0, 25.0)
        # Lower Debt/Equity is better
        de_score = _linear_score(debt_equity, 0.1, 1.5)
        
        return (roe_score * 0.7 + de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Computes growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # Lower PEG (growth relative to price) is better
        peg_score = _linear_score(peg, 0.5, 3.0)
        # Higher ROE also indicates efficient growth/reinvestment capability
        roe_score = _linear_score(roe * 100, 5.0, 25.0)

        return (peg_score * 0.6 + roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results