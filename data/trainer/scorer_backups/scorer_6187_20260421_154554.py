"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to balance with quality/growth
    "quality": 0.25,   # Increased weight to capture stability
    "growth": 0.25,    # Balanced with quality
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
            # We add a small epsilon to avoid log(0) issues in geometric mean logic 
            # if score_val is 0, though we use power-based approach.
            # Using (score/100) allows 0 to be a valid score representing the worst.
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            # The product of (S/100)^w is the core. 
            # If we want to return a value in [0, 100], and sum of weights is not necessarily 1:
            # composite = (product)^(1/sum_weights) * 100
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield are better."""
        # Use a mixture of valuation metrics. 
        # We prioritize PE if available, then PB.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Defaulting to neutral-ish values for missing data
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        div = div if div is not None else 0.0

        # Scoring: Lower PE/PB is better.
        # We use a simple linear clamp for comparison.
        score_pe = _linear_score(1.0 / pe if pe > 0 else 0, 1/30, 1/5) # Scale: PE 5-30
        score_pb = _linear_score(1.0 / pb, 1/3, 1/1) # Scale: PB 1-3
        
        # Dividend yield is a bonus
        score_div = _linear_score(div, 0.05, 0.0) # Scale: 0% to 5%

        return (score_pe * 0.4) + (score_pb * 0.4) + (score_div * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE score (Targeting 15-30%)
        score_roe = _linear_score(roe, 0.30, 0.10)
        # Margin score (Targeting 15-30%)
        score_margin = _linear_score(margin, 0.20, 0.05)
        # Debt score (Lower is better)
        score_debt = _linear_score(1.0 / (debt + 0.01), 1/0.1, 1/2.0)

        return (score_roe * 0.4) + (score_margin * 0.4) + (score_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and better PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 2.0

        # ROE is a proxy for quality-growth
        score_roe = _linear_score(roe, 0.25, 0.05)
        # PEG score (Lower is better)
        score_peg = _linear_score(1.0 / peg, 1/3, 1/0.5)

        return (score_roe * 0.5) + (score_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results