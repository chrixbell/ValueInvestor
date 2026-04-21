"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced weight to allow quality/growth to balance
    "quality": 0.30,   # Increased weight for stability
    "growth": 0.30,    # Increased weight to capture expansion potential
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
            # Using a small epsilon to prevent zero-division/zero-log issues in geometric mean
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
        """Evaluate value using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE is negative or None
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Standard PE range for value investing (e.g., 2 to 30)
            pe_score = _linear_score(pe, 2.0, 30.0)
        elif pe is not None and pe <= 0:
            # Negative PE can be good (profitable) or bad. We treat it as high value if small negative, 
            # but for simplicity, we'll map it to a baseline.
            pe_score = 50.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.5, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 50.0

        # Dividend yield as a value booster
        div_score = 0.0
        if result.valuation.dividend_yield is not None:
            div_score = _linear_score(result.valuation.dividend_yield, 2.0, 8.0)

        return (pe_score * 0.4 + pb_score * 0.4 + div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE score (Targeting 5% to 25%)
        roe_score = _linear_score(roe * 100, 5.0, 25.0)
        
        # Debt-to-Equity (Lower is better, target 0 to 1.0)
        # We invert the logic: high debt = low score
        debt_score = _linear_score(1.0 / (debt_to_equity + 0.01), 0.1, 2.0) if debt_to_equity > 0 else 100.0
        # Alternative: direct linear mapping for debt
        debt_score = _linear_score(0.5, 0.0, 1.5) if debt_to_equity is not None else 50.0

        return (roe_score * 0.7 + debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG score (Lower is better, target 0.5 to 2.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.5)
        else:
            # If PEG is unavailable or negative (high growth/low PE), assign high score but cap it
            peg_score = 70.0

        # ROE is often a proxy for growth sustainability
        roe_score = _linear_score(roe * 100, 5.0, 20.0)

        return (peg_score * 0.6 + roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results