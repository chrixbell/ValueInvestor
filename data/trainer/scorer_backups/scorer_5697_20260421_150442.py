"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Focus on value stability
    "quality": 0.30,   # Increase quality weight to capture fundamental strength
    "growth": 0.20,    # Growth as a secondary driver
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
            # Use a small epsilon to prevent math domain errors with zero scores in geometric mean
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Weighted geometric mean scaled back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use forward PE if available, otherwise trailing
        pe_val = result.valuation.pe_forward if pe is None else pe
        
        # Define reasonable bounds for A-share/HK market
        pe_best, pe_worst = 15.0, 40.0
        pb_best, pb_worst = 1.5, 5.0

        pe_s = _linear_score(pe_val if pe_val is not None else 30.0, pe_best, pe_worst) if pe_val is not None else 50.0
        pb_s = _linear_score(pb, pb_best, pb_worst) if pb is not None else 50.0
        
        # For value, lower is better, so we invert the linear score logic if needed 
        # but _linear_score is (val-worst)/(best-worst). For PE, lower is better.
        # Let's re-map: higher score for lower PE.
        pe_score = 100.0 - pe_s if pe_val is not None else 50.0
        pb_score = 100.0 - pb_s if pb is not None else 50.0
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity

        # ROE: Higher is better
        roe_score = 50.0
        if roe is not None:
            # Target ROE between 5% and 25%
            roe_score = _linear_score(roe, 25.0, 5.0) # Note: if best=25, worst=5
            # Adjusting _linear_score usage: it's (val-worst)/(best-worst). 
            # If we want higher score for higher value, best should be > worst.
            roe_score = _linear_score(roe, 25.0, 5.0) # This is actually wrong if best < worst.
            # Let's use a simpler approach for clarity:
            roe_score = max(0.0, min(100.0, (roe - 5.0) / (25.0 - 5.0) * 100.0)) if roe is not None else 50.0
        
        # Debt/Equity: Lower is better
        debt_score = 50.0
        if debt_eq is not None:
            # Target Debt/Equity between 0 and 1.5
            debt_score = max(0.0, min(100.0, (1.5 - debt_eq) / 1.5 * 100.0))

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG: Lower is better (growth at reasonable price)
        peg_score = 50.0
        if peg is not None:
            # PEG of 1.0 is neutral, 0.5 is great, 2.0 is poor
            peg_score = max(0.0, min(100.0, (2.0 - peg) / 1.5 * 100.0))

        # ROE: Higher is better
        roe_score = 50.0
        if roe is not None:
            roe_score = max(0.0, min(100.0, (roe - 5.0) / 25.0 * 100.0))

        return (peg_score + roe_score) / 2.0