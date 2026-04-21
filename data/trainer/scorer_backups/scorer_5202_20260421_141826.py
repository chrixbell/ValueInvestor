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
    "quality": 0.25,   # Increased to capture stable cash flows and ROE
    "growth": 0.25,    # Increased to capture expansion potential
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

        # We use an additive weighted average instead of geometric mean to prevent 
        # a single zero-score (e.g., an edge case in one metric) from wiping out the 
        # entire composite score, while still allowing for robust ranking.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE as primary driver if available, then fallback to trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or zero/negative PE
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Low PE is better for value
            pe_score = _linear_score(1.0/pe, 1.0/30.0, 1.0/5.0) if pe > 0 else 0.0

        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Low PB is better for value
            pb_score = _linear_score(pb, 1.0, 5.0)
            pb_score = 100.0 - pb_score # Invert: lower PB -> higher score

        # Combine PE and PB scores (weighted equally)
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        d_e = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE score: Higher is better. Using a range of -10% to 30%
        roe_score = _linear_score(roe, 0.30, -0.10)

        # Debt-to-Equity score: Lower is better (lower leverage).
        # If d_e is 0 or negative, it's a perfect score.
        if d_e <= 0:
            de_score = 100.0
        else:
            # Map d_e of 0-2 to a score. High leverage (d_e > 2) gets low score.
            de_score = _linear_score(1.0/d_e, 1.0/2.0, 1.0/0.5)
            de_score = 100.0 - de_score # Invert to make low d/e high score
            # Re-correcting logic: if d_e is 0.5, de_score should be high.
            # Let's use a simpler approach:
            de_score = max(0.0, min(100.0, (2.0 - d_e) / 2.0 * 100.0)) if d_e > 0 else 100.0

        # Re-calculating de_score more robustly
        if d_e <= 0:
            de_score = 100.0
        elif d_e < 2.0:
            de_score = 100.0 - (d_e / 2.0 * 100.0) # This is actually wrong, let's fix
        
        # Let's just use a standard linear clamp for de_score: 0 debt = 100, 2.0 debt = 0
        de_score = max(0.0, min(100.0, (2.0 - d_e) / 2.0 * 100.0)) if d_e > 0 else 100.0
        # Wait, the logic above is: if d_e=0.5 -> (1.5/2)*100 = 75. If d_e=2 -> 0. Correct.
        # If d_e=0 -> 100.

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio

        # ROE is a proxy for growth/efficiency
        roe_score = _linear_score(roe, 0.25, -0.10)

        if peg is None or peg <= 0:
            # If PEG is unavailable, we rely on ROE. We treat high positive/missing PEG as neutral-low
            peg_score = 50.0
        else:
            # Lower PEG is better (growth at reasonable price)
            peg_score = _linear_score(1.0/peg, 1.0/0.5, 1.0/3.0)
            # If PEG is very low (e.g. 0.5), 1/0.5=2, 1/3=0.33 -> (2-0.33)/(2-0.33)?? No.
            # Let's use: peg_score = max(0, min(100, (3.0 - peg) / 3.0 * 100))
            peg_score = max(0.0, min(100.0, (3.0 - peg) / 3.0 * 100.0))

        return (roe_score + peg_score) / 2.0