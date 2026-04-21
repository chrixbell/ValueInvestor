"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more focus on quality/growth
    "quality": 0.25,   # Increased to prioritize stable companies
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
            # Use a small epsilon to prevent zero-out in geometric mean if one factor is 0
            # but allow it to be very low.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores [0, 1] scaled back to [0, 100]
            # Note: product_of_powers is already (S1/100)^w1 * (S2/100)^w2...
            # If weights sum to 1, product_of_powers is the result. 
            # To handle non-unit weights, we use the power-based normalization.
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

        # Fallback if both are None
        if pe is None and pb is None:
            return 50.0

        # Scoring logic: lower PE/PB is better
        # We use a wider range for normalization to avoid extreme sensitivity
        pe_score = 0.0
        if pe is not None:
            # If PE is negative (loss), it's still "cheap" but riskier. 
            # For simplicity in this iteration, we treat negative PE as very high (bad) or neutral.
            # Using a clamp: PE < 0 is treated as high, but we use absolute for comparison if needed.
            # Here: PE 0-5 -> 100, PE 30+ -> 0
            if pe > 0:
                pe_score = max(0.0, min(100.0, (35.0 - pe) / 35.0 * 100.0))
            else:
                pe_score = 20.0 # Low score for negative PE

        pb_score = 0.0
        if pb is not None:
            if pb > 0:
                pb_score = max(0.0, min(100.0, (10.0 - pb) / 10.0 * 100.0))
            else:
                pb_score = 50.0

        # If only one is available, return that; otherwise average
        if pe is None: return pb_score
        if pb is None: return pe_score
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE (Higher is better)
        roe_score = 50.0
        if roe is not None:
            # Normalize ROE (assume 20% is great, -10% is bad)
            roe_score = max(0.0, min(100.0, (roe + 0.1) / 0.3 * 100.0))

        # Debt-to-Equity (Lower is better)
        de_score = 50.0
        if debt_equity is not None:
            # Normalize DE (assume 1.0 is neutral, 3.0 is bad)
            de_score = max(0.0, min(100.0, (2.0 - debt_equity) / 2.0 * 100.0))

        if roe is None: return de_score
        if debt_equity is None: return roe_score
        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Net Margin."""
        peg = result.valuation.peg_ratio
        margin = result.financials.net_margin

        # PEG (Lower is better, but avoid division by zero or negative)
        peg_score = 50.0
        if peg is not None:
            if peg > 0:
                peg_score = max(0.0, min(100.0, (3.0 - peg) / 3.0 * 100.0))
            else:
                peg_score = 70.0 # Negative PEG (negative earnings) is tricky; treat as high growth potential

        # Net Margin (Higher is better)
        margin_score = 50.0
        if margin is not None:
            # Assume 20% is great, -5% is bad
            margin_score = max(0.0, min(100.0, (margin + 0.05) / 0.25 * 100.0))

        if peg is None: return margin_score
        if margin is None: return peg_score
        return (peg_score + margin_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results