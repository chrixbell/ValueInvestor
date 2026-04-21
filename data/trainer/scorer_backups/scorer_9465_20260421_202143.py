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
    "quality": 0.25,   # Increased to capture fundamental stability
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
            # Using (score_val / 100.0) for geometric mean to prevent one factor from dominating
            # and ensure a zero score in any category significantly impacts the total.
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            # The product of powers already accounts for weights. 
            # If we use (S/100)^w, the result is in [0, 1].
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score focusing on low multiples."""
        # Use PE and PB as primary drivers. Handle None values.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Fallback to 20 if PE is None (assumes a neutral-to-high PE)
        pe_val = pe if pe is not None and pe > 0 else 20.0
        pb_val = pb if pb is not None and pb > 0 else 2.0
        ps_val = ps if ps is not None and ps > 0 else 2.0

        # Score components (lower is better for value)
        # We use fixed reasonable bounds to map to [0, 100]
        s_pe = _linear_score(20.0, 5.0, 40.0) if pe_val > 0 else 50.0
        s_pb = _linear_score(2.0, 0.5, 10.0) if pb_val > 0 else 50.0
        s_ps = _linear_score(2.0, 0.5, 10.0) if ps_val > 0 else 50.0
        
        # Simple average of value metrics
        return (s_pe + s_pb + s_ps) / 3.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score focusing on profitability and solvency."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE: High is good (Targeting 20% as best, 0% as worst)
        s_roe = _linear_score(roe * 100, 0.0, 30.0)
        # Margin: High is good (Targeting 15% as best, -5% as worst)
        s_margin = _linear_score(margin * 100, -5.0, 20.0)
        # Debt: Low is good (Targeting 30% as best, 150% as worst)
        s_debt = _linear_score(debt * 100, 30.0, 150.0)
        # Invert debt score because lower is better
        s_debt = 100.0 - s_debt

        return (s_roe + s_margin + s_debt) / 3.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score focusing on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 1.5

        # ROE is a proxy for internal growth capacity
        s_roe = _linear_score(roe * 100, 0.0, 30.0)
        # PEG: Low is better (Targeting 0.5 as best, 3.0 as worst)
        s_peg = _linear_score(peg, 0.5, 3.0)
        # Invert PEG score because lower is better
        s_peg = 100.0 - s_peg

        return (s_roe + s_peg) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results