"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow room for quality/growth
    "quality": 0.25,   # Increased to prioritize stable companies
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
            # We use a small epsilon to prevent log(0) issues in geometric mean if score_val is 0
            # but since we divide by 100, it's already [0, 1].
            normalized_score = score_val / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers already effectively incorporates the weights.
            # (S1^w1 * S2^w2) is already the correct form for a weighted geometric mean 
            # if total_weight is normalized to 1. We adjust for non-normalized weights.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None values
        if pe is None or pb is None:
            return 50.0

        # PE/PB scoring (lower is better)
        # We use a wide range for clamping to capture meaningful differences.
        pe_score = _linear_score(1.0 / (pe + 1e-6), 30.0, 1.0) if pe > 0 else 0.0
        pb_score = _linear_score(1.0 / (pb + 1e-6), 5.0, 0.1) if pb > 0 else 0.0
        
        # Dividend yield is a great value component
        div_score = 0.0
        if result.valuation.dividend_yield is not None:
            div_score = _linear_score(result.valuation.dividend_yield, 10.0, 0.0)

        return (pe_score * 0.4 + pb_score * 0.4 + div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE scoring (higher is better)
        roe_score = _linear_score(roe, 0.25, -0.1)
        # Leverage scoring (lower is better)
        leverage_score = _linear_score(1.0 / (debt_equity + 1e-6), 2.0, 0.1) if debt_equity > 0 else 100.0
        # Margin scoring (higher is better)
        margin_score = _linear_score(margin, 0.20, -0.1)

        return (roe_score * 0.5 + leverage_score * 0.3 + margin_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and revenue growth proxy."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # If PEG is available, it's a powerful growth-at-reasonable-price metric
        if peg is not None and peg > 0:
            peg_score = _linear_score(1.0 / peg, 2.0, 0.1)
        else:
            # Fallback to ROE-based growth if PEG is missing/invalid
            peg_score = _linear_score(roe, 0.30, -0.1)

        # Combine with ROE for a growth-quality hybrid
        return (peg_score * 0.7 + _linear_score(roe, 0.20, -0.1) * 0.3)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results