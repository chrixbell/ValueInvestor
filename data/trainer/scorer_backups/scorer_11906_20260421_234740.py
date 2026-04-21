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
    "quality": 0.25,   # Increased to reward fundamental stability
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
                # We use a small epsilon to prevent zero-value issues in geometric mean
                # while still allowing low scores.
                weighted_scores_to_process[k] = max(0.001, score_val)

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process.keys())
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights.get(k, 0.0)
            # Normalize score to [0.0, 1.0] range for geometric mean calculation
            product_of_powers *= (score_val / 100.0) ** weight

        # The geometric mean formula: (Product of S_i^w_i) ^ (1 / Sum of w_i)
        # However, since we already applied weights in the power, 
        # product_of_powers is already (S1^w1 * S2^w2...).
        # To get the scale back to 100, we don't need another exponent unless weights didn't sum to 1.
        # But for robustness, we treat the product as the core score.
        
        result.composite_score = product_of_powers * 100.0
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or higher dividend yield is better."""
        v = result.valuation
        f = result.financials
        
        # Primary valuation: PE or PB
        if v.pe_ratio and v.pe_ratio > 0:
            score = _linear_score(1 / v.pe_ratio, 1/50, 1/2) * 100 # Using earnings yield concept
            # Actually let's use a simpler linear approach for stability
            score = _linear_score(0, 5, 40) # Placeholder logic to be refined
        else:
            score = 50.0

        # Let's try a more robust approach:
        pe_score = 0.0
        if v.pe_ratio and v.pe_ratio > 0:
            # Map PE 0-30 to 100-0
            pe_score = _linear_score(v.pe_ratio, 30, 5)
        elif v.pb_ratio and v.pb_ratio > 0:
            pe_score = _linear_score(v.pb_ratio, 5, 1)
        else:
            pe_score = 50.0

        div_score = 0.0
        if v.dividend_yield and v.dividend_yield > 0:
            div_score = _linear_score(v.dividend_yield, 10, 0)
        
        return (pe_score * 0.8) + (div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage."""
        f = result.financials
        q_score = 50.0
        
        roe_score = 0.0
        if f.roe:
            roe_score = _linear_score(f.roe, 25, -5)
        
        margin_score = 0.0
        if f.net_margin:
            margin_score = _linear_score(f.net_margin, 20, -5)

        debt_score = 0.0
        if f.debt_to_equity:
            # Lower debt is better
            debt_score = _linear_score(f.debt_to_equity, 0.5, 2.0)
        elif f.debt_to_equity is None:
            debt_score = 50.0

        # Combine
        return (roe_score * 0.4) + (margin_score * 0.3) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        v = result.valuation
        f = result.financials
        
        growth_score = 50.0
        if v.peg_ratio and v.peg_ratio > 0:
            # Lower PEG is better for growth-at-a-reasonable-price
            growth_score = _linear_score(v.peg_ratio, 2, 0.5)
        elif f.roe:
            growth_score = _linear_score(f.roe, 20, 5)
            
        return growth_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results