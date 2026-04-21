"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced weight to allow quality/growth more room
    "quality": 0.25,   # Increased to prioritize stronger balance sheets/profitability
    "growth": 0.25,    # Balanced weight for growth potential
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Since we are using product of powers, the result is already normalized.
            # (S1^w1 * S2^w2) results in a value where if weights sum to 1, it's the geometric mean.
            # If they don't sum to 1, we adjust by dividing the exponent by total_weight.
            # However, the standard way to do weighted geometric mean is: (Π S_i^w_i)^(1 / Σ w_i)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use pe_forward if available, else pe_ratio. 
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or negative PE (loss making)
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 25] for value
            pe_score = _linear_score(pe, 1.0, 25.0)
        elif pe is not None and pe <= 0:
            # Negative PE is technically "cheap" but risky. Assign a baseline score.
            pe_score = 10.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            # Target PB range [0.5, 3]
            pb_score = _linear_score(pb, 0.5, 3.0)
        elif pb is not None and pb <= 0:
            pb_score = 5.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better (Target 5% to 30%)
        roe_score = _linear_score(roe * 100, 5.0, 30.0) if roe is not None else 0.0
        
        # Debt: Lower is better (Target 0 to 1.0)
        # We invert the logic: low debt = high score
        debt_score = _linear_score(0.5, 0.0, 1.5) if debt_to_equity is not None else 0.0

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and Gross Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        gross_margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0

        # PEG: Lower is better (Target 0.5 to 2.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.0)
        elif peg is not None and peg <= 0:
            peg_score = 20.0 # Very low/negative PEG is attractive for growth
        else:
            peg_score = 50.0

        # Gross Margin: Higher is better (Target 10% to 40%)
        gm_score = _linear_score(gross_margin * 100, 10.0, 40.0) if gross_margin is not None else 0.0

        return (peg_score + gm_score) / 2.0