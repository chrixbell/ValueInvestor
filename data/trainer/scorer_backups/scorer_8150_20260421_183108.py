"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced to allow more weight for quality/growth
    "quality": 0.30,   # Increased to capture stable returns
    "growth": 0.30,    # Increased to capture expansion potential
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
            # Use a small epsilon to prevent issues with log/zero in geometric mean logic
            # though we scale by 100, (score_val / 100.0) is in [0, 1]
            normalized_val = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of (val/100)^weight is already the weighted geometric mean 
            # if weights are normalized. We scale back by 100.
            # Note: (x^w1 * y^w2) is not the same as (x*y)^(avg_weight).
            # Correct geometric mean of normalized values is: product^(1/sum_weights)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE might be negative or zero
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 25] for a good value score
            pe_score = _linear_score(pe, 1.0, 25.0)
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; often indicates losses. Give it a low but non-zero score
            pe_score = 10.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            # Target PB range [0.5, 3.0]
            pb_score = _linear_score(pb, 0.5, 3.0)
        elif pb is not None and pb <= 0:
            pb_score = 5.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # Higher ROE is better
        roe_score = _linear_score(roe, 0.05, 0.25) # 5% to 25%
        
        # Lower Debt-to-Equity is better
        # We invert the logic: lower debt = higher score
        # If debt_equity is 0.5, it's "best", if 2.0 it's "worst"
        debt_score = _linear_score(1.0 / (debt_equity + 0.01), 0.5, 2.0) if debt_equity > 0 else 100.0
        if debt_equity <= 0: debt_score = 100.0

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0

        # ROE as a proxy for internal growth capability
        roe_score = _linear_score(roe, 0.10, 0.30)

        # PEG: lower is better (growth adjusted valuation)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.5)
        else:
            peg_score = 50.0

        return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results