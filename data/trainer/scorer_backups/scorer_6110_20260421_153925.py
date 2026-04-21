"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased to emphasize stability and profitability
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

        # Momentum is a placeholder
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
            # Add a tiny epsilon to avoid zero-score issues in geometric mean
            normalized_score = (max(0.0, score_val) + 1e-6) / 100.0
            product_of_powers *= normalized_score ** weight

        if total_weight > 0:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        scores = []
        # Use Forward PE if available, else trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        if pe is not None and pe > 0:
            # Lower PE is better. Clamp range [1, 40]
            scores.append(_linear_score(pe, 1.0, 40.0))
        elif pe is not None and pe <= 0:
            # Negative PE (profitable) is good, but we treat as "best" in this scale
            scores.append(100.0)
        else:
            scores.append(50.0)

        if pb is not None and pb > 0:
            scores.append(_linear_score(pb, 0.5, 10.0))
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        scores = []
        roe = result.financials.roe if result.financials.roe is not None else None
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else None

        if roe is not None:
            # Higher ROE is better. Clamp range [-20%, 40%]
            score = (roe + 20) / (40 + 20) * 100
            scores.append(max(0.0, min(100.0, score)))
        else:
            scores.append(50.0)

        if debt_equity is not None:
            # Lower Debt/Equity is better. Clamp range [0, 2]
            score = (2.0 - debt_equity) / 2.0 * 100
            scores.append(max(0.0, min(100.0, score)))
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Net Margin."""
        scores = []
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.net_margin if result.financials.net_margin is not None else None

        if peg is not None and peg > 0:
            # Lower PEG is better. Clamp range [0.5, 3.0]
            score = (3.0 - peg) / (3.0 - 0.5) * 100
            scores.append(max(0.0, min(100.0, score)))
        elif peg is not None and peg <= 0:
            scores.append(100.0)
        else:
            scores.append(50.0)

        if margin is not None:
            # Higher Net Margin is better. Clamp range [-5%, 30%]
            score = (margin + 0.05) / (0.30 + 0.05) * 100
            scores.append(max(0.0, min(100.0, score)))
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0