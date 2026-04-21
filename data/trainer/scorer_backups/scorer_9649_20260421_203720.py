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
    "quality": 0.25,   # Increased to reward stability and profitability
    "growth": 0.25,    # Increased to capture expansionary potential
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
            # Using a small epsilon to prevent log(0) in geometric mean logic if score is 0
            # We map [0, 100] to [epsilon, 1.0]
            normalized_score = max(0.0001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The weighted geometric mean formula is (Product of S_i^w_i) ^ (1 / Sum w_i)
            # Since we are multiplying by weights inside the power, we need to handle normalization.
            # However, if product_of_powers is already (S1^w1 * S2^w2...), then the 
            # resulting score is already scaled. We just need to adjust for the sum of weights.
            # If w1+w2 = 1, product_of_powers is the result.
            # If sum(weights) != 1, we need to adjust.
            
            # Re-calculating composite_score correctly for any weight sum:
            # We want Result = (S1^w1 * S2^w2 ...)^(1 / total_weight)
            # But we used normalized_score (0 to 1). We scale it back to 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluate value based on PE and PB ratios."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None or invalid PE (negative/zero)
        if pe is None or pe <= 0:
            # If PE is invalid, rely on PB if available, otherwise neutral
            if pb is not None and pb > 0:
                return _linear_score(pb, 1.0, 10.0)
            return 50.0

        # Score for PE (lower is better)
        pe_score = _linear_score(pe, 5.0, 30.0)
        
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1.0, 5.0)
            return (pe_score + pb_score) / 2.0
        
        return pe_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality based on ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE Score: higher is better (normalized around typical ranges)
        roe_score = _linear_score(roe, 0.05, 0.25)
        
        # Debt Score: lower is better (clamped at 0 for high leverage)
        debt_score = _linear_score(debt_equity, 0.0, 1.0)

        return (roe_score * 0.7 + debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth based on PEG and Net Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.1

        # PEG Score: lower is better (but we clamp to avoid negative scores)
        peg_score = _linear_score(peg, 0.5, 2.0)
        
        # Margin Score: higher is better
        margin_score = _linear_score(margin, 0.05, 0.30)

        return (peg_score * 0.5 + margin_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[Screening_Rank]:
        """Sorts all results by composite score and assigns ranks."""
        # Note: The return type in the original snippet was not fully defined, 
        # but we maintain the interface.
        for r in results:
            self.score(r)
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            # Using 1-based ranking
            res.rank = i + 1
        return sorted_results