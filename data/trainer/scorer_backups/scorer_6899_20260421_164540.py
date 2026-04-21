"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Balanced with quality for a more robust multi-factor approach
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to avoid log(0) issues during geometric mean if score is 0
            # However, since we divide by 100, (score/100) is in [0, 1].
            # We use max(score_val, 0.0001) to ensure stability.
            safe_score = max(score_val, 0.0001) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # The product of (score/100)^weight is essentially the geometric mean 
            # if we normalize by total_weight later.
            # Since sum(weights) might not be 1.0, we adjust the exponent.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use forward PE if available, else trailing PE.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE is negative or None
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Low PE is better. We'll use a range of 0 to 30 for PE and 0 to 5 for PB.
            pe_score = _linear_score(pe, 30.0, 0.0)
        elif pe is not None and pe <= 0:
            # Negative PE can be tricky; assume it's a "value" trap or high-growth/loss scenario.
            # For simplicity, score it low but not zero.
            pe_score = 10.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 5.0, 0.0)
        elif pb is not None and pb <= 0:
            pb_score = 10.0

        # Dividend yield is also a value metric
        div_yield = result.valuation.dividend_yield
        div_score = 0.0
        if div_yield is not None:
            # High dividend yield is good (e.g., 0% to 10%)
            div_score = _linear_score(div_yield, 8.0, 0.0)

        # Weighting sub-value components
        return (pe_score * 0.4 + pb_score * 0.3 + div_score * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE Score: Higher is better (range -20% to 40%)
        roe_score = _linear_score(roe * 100, 40.0, -20.0)
        
        # Leverage Score: Lower is better (range 0 to 2.0)
        debt_score = _linear_score(debt_equity, 1.0, 0.0)
        
        # Margin Score: Higher is better (range -10% to 30%)
        margin_score = _linear_score(margin * 100, 30.0, -10.0)

        return (roe_score * 0.5 + debt_score * 0.2 + margin_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio

        # ROE is a proxy for growth potential/efficiency
        roe_score = _linear_score(roe * 100, 30.0, -10.0)

        # PEG: Lower is better (range 0 to 2.5). 
        peg_score = 0.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 2.5, 0.1)
        elif peg is not None and peg <= 0:
            # Negative PEG often means negative earnings but high growth potential.
            # We treat it as a neutral/high score for growth.
            peg_score = 80.0

        return (roe_score * 0.4 + peg_score * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results