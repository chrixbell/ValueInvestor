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
    "quality": 0.25,   # Increased to reward stable profitability
    "growth": 0.25,    # Increased to capture expansionary potential
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
            # Normalize score to [0.0, 1.0] range for geometric mean calculation
            # We use a small epsilon to prevent log(0) if using geometric mean logic via power
            norm_score = max(0.0, score_val) / 100.0
            product_of_powers *= (norm_score ** weight)

        if total_weight > 0:
            # The product_of_powers is (S1/100)^w1 * (S2/100)^w2 ...
            # To get the weighted geometric mean on [0, 100], we scale it.
            # Since product_of_powers is already weighted, we don't need (1/total_weight) 
            # unless the weights themselves aren't normalized to 1.0.
            # If they aren't, we normalize the result.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Score based on valuation metrics."""
        # Use PE or PB depending on availability, fallback to PS.
        # We want low valuation (low PE/PB) to result in high score.
        v_pe = result.valuation.pe_ratio if result.valuation.pe_ratio is not None and result.valuation.pe_ratio > 0 else None
        v_pb = result.valuation.pb_ratio if result.valuation.pb_ratio is not None and result.valuation.pb_ratio > 0 else None
        v_ps = result.valuation.ps_ratio if result.valuation.ps_ratio is not None and result.valuation.ps_ratio > 0 else None

        if v_pe:
            return _linear_score(v_pe, 40.0, 5.0)  # Target PE range: 5 to 40
        elif v_pb:
            return _linear_score(v_pb, 4.0, 1.0)   # Target PB range: 1 to 4
        elif v_ps:
            return _linear_score(v_ps, 3.0, 0.5)   # Target PS range: 0.5 to 3
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Score based on profitability and solvency."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE is a primary quality metric.
        # Debt to equity: lower is better. (Assume 1.0 is neutral)
        q_roe = _linear_score(roe * 100, 25.0, 0.0) # Target ROE: 0% to 25%
        q_margin = _linear_score(margin * 100, 20.0, 0.0) # Target Margin: 0% to 20%
        q_debt = _linear_score(debt, 0.5, 2.0) # Target Debt: 0.5 to 2.0
        
        # Combine quality metrics (weighted average of sub-scores)
        return (q_roe * 0.5 + q_margin * 0.3 + q_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Score based on growth potential and valuation efficiency."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else None
        
        # If PEG is available, it's a strong indicator of growth-at-reasonable-price.
        if peg is not None:
            # PEG of 1.0 is fair, < 1 is great.
            return _linear_score(peg, 0.5, 3.0)
        else:
            # Fallback to ROE as a proxy for growth/efficiency
            return _linear_score(roe * 100, 30.0, 0.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        # Sort descending: highest score is rank 1.
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results