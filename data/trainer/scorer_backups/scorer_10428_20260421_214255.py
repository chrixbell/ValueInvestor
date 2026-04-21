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
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Increased growth weight to capture upside potential
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
            # We use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # However, we allow the zero to propagate as it represents a fundamental failure.
            normalized_score = score_val / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The formula (product_of_powers ** (1.0 / total_weight)) is mathematically
            # equivalent to the weighted geometric mean if we treat product_of_powers 
            # as (S1^w1 * S2^w2...). Since the weights are already normalized in 
            # product_of_powers, we just scale back.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or zero/negative values for PE
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Lower PE is better. We use a range of 0-40 for the 'best' side to avoid extreme outliers.
            pe_score = _linear_score(pe, best=5.0, worst=40.0)

        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Lower PB is better.
            pb_score = _linear_score(pb, best=1.0, worst=10.0)

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # ROE: Higher is better (using log scale for growth-like qualities)
        # We assume ROE is expressed as a decimal (e.g., 0.15 for 15%) or percentage.
        # We handle the scale by assuming typical ROE ranges.
        if roe > 0:
            roe_score = _log_score(roe, best=0.25, worst=0.01)
        else:
            roe_score = 0.0

        # Debt/Equity: Lower is better.
        if debt_to_equity <= 0:
            de_score = 100.0
        else:
            de_score = _linear_score(debt_to_equity, best=0.2, worst=2.0)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        gross_margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0

        # PEG: Lower is better (ideal is around 1.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, best=0.5, worst=3.0)
        else:
            # If PEG is unavailable, we weight the gross margin more heavily or use a neutral 50
            peg_score = 50.0

        # Gross Margin: Higher is better
        if gross_margin > 0:
            gm_score = _log_score(gross_margin, best=0.40, worst=0.05)
        else:
            gm_score = 0.0

        return (peg_score + gm_score) / 2.0