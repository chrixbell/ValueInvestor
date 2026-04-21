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
    "quality": 0.25,   # Increased to capture stability
    "growth": 0.25,    # Increased to capture upside potential
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

        # Using additive weighted mean for better stability when dealing with zero-scores in geometric means.
        # In financial scoring, a single bad factor (like 0 dividend yield) shouldn't necessarily zero out the whole score.
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
        """Calculate value score based on PE and PB."""
        # Use forward PE if available, else trailing
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or zero/negative PE
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range for value: low is better. 
            # We use a soft clamp: 1-25 is good, >40 is bad.
            pe_score = _linear_score(pe, 1.0, 40.0)
        elif pe is not None and pe <= 0:
            # Negative PE can be tricky. If it's a loss, it might be undervalued or risky.
            # For simplicity in this factor, we treat negative PE as low value score (but not 0 if it's a massive loss)
            pe_score = 20.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.5, 10.0)
        elif pb is not None and pb <= 0:
            pb_score = 10.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better. Scale 0% to 30%.
        roe_score = _linear_score(roe * 100, 5.0, 30.0) if roe is not None else 0.0
        
        # Debt/Equity: Lower is better. Scale 0 to 1 (100%).
        # If debt/equity is very high, score is 0.
        de_score = _linear_score(debt_equity, 0.1, 2.0) if debt_equity is not None else 50.0
        # Invert de_score because lower debt should be higher score
        de_score = 100.0 - de_score

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and gross margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.gross_margin if result.financials.gross_margin is not None else None

        # PEG: Lower is better (Growth relative to value).
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 3.0)
        else:
            peg_score = 50.0

        # Gross Margin: Higher is better.
        if margin is not None:
            margin_score = _linear_score(margin * 100, 10.0, 50.0)
        else:
            margin_score = 50.0

        return (peg_score + margin_score) / 2.0