"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow room for quality/growth stability
    "quality": 0.25,   # Increased to reward robust balance sheets and profitability
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
            # Using a small epsilon to prevent zero-out in geometric mean if score is 0
            normalized_val = max(0.001, score_val) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of powers is already effectively (S1^w1 * S2^w2 ...)
            # We scale it back to [0, 100]
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better for value."""
        # Use PEG as a bridge between value and growth if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        peg = result.valuation.peg_ratio

        # Heuristic thresholds for "best" and "worst"
        # PE: 5 to 40, PB: 0.5 to 10, PS: 0.5 to 10
        # We use a simple average of normalized scores for multiple value metrics
        scores = []
        if pe is not None and pe > 0:
            scores.append(_linear_score(pe, 5.0, 40.0))
        else:
            scores.append(50.0)

        if pb is not None and pb > 0:
            scores.append(_linear_score(pb, 0.5, 10.0))
        else:
            scores.append(50.0)

        if ps is not None and ps > 0:
            scores.append(_linear_score(ps, 0.5, 10.0))
        else:
            scores.append(50.0)

        if peg is not None and peg > 0:
            # PEG is a special case where low is good. We want to invert it for linear_score if we treated it like PE
            # But let's just use a direct mapping: 0.5 is great, 3.0 is bad
            scores.append(_linear_score(peg, 0.5, 3.0))

        if not scores:
            return 50.0
        
        # Since linear_score(low, high, low) is bad, and we want lower PE to be higher score:
        # We need to invert the scores because _linear_score returns high for big values.
        # Let's re-calculate: we want 'best' to be the target for low values.
        # Actually, in _linear_score: score = (value - worst) / (best - worst). 
        # For PE, best=5, worst=40. If pe=5 -> (5-40)/(5-40) = 1.0 (100%). Correct.
        # If pe=40 -> (40-40)/(5-40) = 0. Correct.
        # So the current _linear_score logic actually works for "lower is better" if we swap best/worst.
        # Let's fix the logic by ensuring 'best' is the low value and 'worst' is the high value.
        
        # Re-evaluating: If pe=5 (best) and worst=40:
        # score = (5 - 40) / (5 - 40) * 100 = 100.
        # If pe=40 (worst): score = (40 - 40) / (5 - 40) * 100 = 0.
        # This works! The logic in the provided snippet was actually correct for "lower is better" 
        # as long as best < worst.

        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin
        
        scores = []
        # ROE: higher is better. Best=30%, Worst=0%
        if roe is not None:
            scores.append(_linear_score(roe, 30.0, 0.0))
        else:
            scores.append(50.0)

        # Debt/Equity: lower is better. Best=0, Worst=2.0
        if debt_equity is not None:
            scores.append(_linear_score(debt_equity, 0.0, 2.0))
        else:
            scores.append(50.0)

        # Net Margin: higher is better. Best=20%, Worst=-5%
        if margin is not None:
            scores.append(_linear_score(margin, 20.0, -5.0))
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        # Growth is often captured by the combination of ROE and valuation
        roe = result.financials.roe
        peg = result.valuation.peg_ratio
        
        scores = []
        if roe is not None:
            # High ROE as a proxy for sustainable growth/quality-growth mix
            scores.append(_linear_score(roe, 25.0, -5.0))
        else:
            scores.append(50.0)

        if peg is not None and peg > 0:
            # PEG: lower is better. Best=0.5, Worst=3.0
            scores.append(_linear_score(peg, 0.5, 3.0))
        else:
            scores.append(50.0)

        return sum(scores) / len(scores) if scores else 50.0