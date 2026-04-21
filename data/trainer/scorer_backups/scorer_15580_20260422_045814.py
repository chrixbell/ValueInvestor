"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to balance with quality/growth
    "quality": 0.25,   # Increased quality weight to ensure stability
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent 0.0 from zeroing out the geometric mean
            # if one factor is unexpectedly bad, while still penalizing it.
            safe_score = max(0.001, score_val) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # The geometric mean formula is (product of scores^weights)^(1/sum_of_weights)
            # Here, since we already applied weights in the loop:
            # product_of_powers = S1^w1 * S2^w2 ...
            # The result is scaled back to [0, 100]
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB."""
        # Use PE if available, else PB. If neither, neutral 50.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        if pe is not None and pe > 0:
            # PE-based scoring (lower is better)
            return _linear_score(1, pe, 60) # Typical range: 1 to 60
        elif pb is not None and pb > 0:
            # PB-based scoring (lower is better)
            return _linear_score(1, pb, 10)
        
        # Fallback to dividend yield if PE/PB are missing or invalid
        dy = result.valuation.dividend_yield
        if dy is not None and dy > 0:
            return _linear_score(dy, 0.01, 0.10)
            
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        d_e = result.financials.debt_to_equity

        # Score components
        roe_score = 0.0
        de_score = 0.0

        if roe is not None:
            # Higher ROE is better. Range [-0.2, 0.4]
            roe_score = _linear_score(roe, 0.40, -0.20)
        else:
            roe_score = 50.0

        if d_e is not None:
            # Lower Debt/Equity is better. Range [0, 2.0]
            de_score = _linear_score(1/max(0.001, d_e), 1.0, 0.1)
        else:
            de_score = 50.0

        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG-based scoring (lower is better)
        if peg is not None and peg > 0:
            peg_score = _linear_score(1, peg, 3.0)
        elif roe is not None:
            # If no PEG, use ROE as a proxy for growth potential
            peg_score = _linear_score(roe, 0.30, -0.10)
        else:
            peg_score = 50.0

        return peg_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all stocks based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks (1 to N)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results