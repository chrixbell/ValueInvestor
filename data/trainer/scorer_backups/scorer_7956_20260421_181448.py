"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow for higher quality/growth contribution
    "quality": 0.25,   # Increased to capture stability and profitability
    "growth": 0.25,    # Balanced with quality to find profitable growth
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

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0
        
        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


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

        # We use a slightly modified geometric mean approach. 
        # To prevent a single zero score from wiping out the entire composite,
        # we use (score + epsilon) to ensure a small positive value.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use (score + epsilon) / 100 to allow for non-zero products even if score is 0
            normalized_val = (score_val + epsilon) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The formula simplifies to (product_of_powers) * 100 if the weights sum to 1.
            # If weights don't sum to 1, we normalize the power.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB → higher score."""
        # Use forward PE as primary, fallback to trailing PE, then PB.
        pe = result.valuation.pe_forward or result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Determine which metric to use
        if pe is not None and pe > 0:
            # Using a reasonable range for PE scoring (e.g., 1 to 40)
            # A PE of 1 is great, 40 is poor.
            return _linear_score(pe, 1.0, 40.0) if pe > 0 else 0.0
        elif pb is not None and pb > 0:
            return _linear_score(pb, 0.5, 10.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe or 0.0
        margin = result.financials.net_margin or 0.0
        debt_equity = result.financials.debt_to_equity or 0.0 # Assuming this is Debt/Equity ratio

        # Quality score components
        score_roe = _linear_score(roe, 0.15, -0.10) # 15% is good, -10% is bad
        score_margin = _linear_score(margin, 0.15, -0.05)
        
        # Debt/Equity: Lower is better. 1.0 (100%) is neutral, >2.0 is bad.
        if debt_equity > 0:
            score_debt = _linear_score(1.0 / debt_equity, 0.1, 5.0) # Inverted to make higher better
        else:
            score_debt = 100.0 # No debt is perfect

        # Combine via simple average
        return (score_roe + score_margin + score_debt) / 3.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG → higher score."""
        peg = result.valuation.peg_ratio
        # Use ROE as a proxy for growth/efficiency if PEG is missing
        roe = result.financials.roe or 0.0

        if peg is not None and peg > 0:
            # PEG of 0.5 is great, 2.0 is poor.
            return _linear_score(1.0 / peg, 0.5, 2.5)
        else:
            # Fallback to ROE-based growth proxy
            return _linear_score(roe, 0.20, -0.10)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results