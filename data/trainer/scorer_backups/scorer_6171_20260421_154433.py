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
    "quality": 0.25,   # Increased quality weight to stabilize the portfolio
    "growth": 0.25,    # Balanced growth contribution
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
            # Use a small epsilon to prevent zero-multiplication issues in geometric mean
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Using forward PE as primary, fallback to trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE might be negative or None
        if pe is None or pe <= 0:
            pe_score = 10.0  # Low penalty for negative PE, but not the best
        else:
            # Target range: PE between 5 and 20 is good.
            pe_score = _linear_score(pe, 5.0, 25.0)

        pb_score = 50.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.5, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 100.0 # Very low PB is good for value

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        d_e = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # ROE score: higher is better (target 5% to 30%)
        roe_score = _linear_score(roe * 100, 5.0, 30.0)
        
        # Debt score: lower is better (target 0 to 1.0)
        # We invert it so higher score = lower debt
        debt_score = _linear_score(d_e, 0.0, 1.5)
        debt_score = 100.0 - debt_score

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Net Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # PEG score: lower is better (target 0.5 to 2.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.5)
        elif peg is not None and peg <= 0:
            peg_score = 100.0
        else:
            peg_score = 50.0

        # Margin score (growth/profitability proxy): higher is better
        margin_score = _linear_score(margin * 100, 5.0, 20.0)

        return (peg_score * 0.5) + (margin_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results