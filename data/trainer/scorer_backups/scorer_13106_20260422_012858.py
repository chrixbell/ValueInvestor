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
    "quality": 0.25,   # Increased to capture more stability and profitability
    "growth": 0.25,    # Balanced with quality to capture high-performing companies
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
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
            # Use a small epsilon to prevent zeroing out the entire score if one factor is 0
            # but allow it to penalize heavily.
            normalized_val = max(score_val / 100.0, 0.0001)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean calculation remains. 
            # If we want the result in [0, 100], and weights sum to 1 (or total_weight),
            # the product of (S/100)^w should be multiplied by 100.
            # Example: (0.5^0.6 * 0.5^0.4) = 0.5. Result: 0.5 * 100 = 50.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or high dividend yield is better."""
        # Use a mix of valuation metrics. Priority: PE, then PB/Yield if PE is missing.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Fallback values for normalization anchors
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0
        dy_best, dy_worst = 5.0, 0.0

        score = 0.0
        if pe and pe > 0:
            # Use log scale for PE to prevent extreme values from dominating
            score = _log_score(pe, pe_best, pe_worst) # Wait, log_score returns high for high. 
            # Correcting logic: we want low PE to be high score.
            # Let's use linear for simplicity or invert the inputs.
            score = _linear_score(pe, pe_best, pe_worst) # This is wrong if best < worst
            # Re-implementing: Higher score for lower PE.
            score = (1.0 - _linear_score(pe, pe_best, pe_worst) / 100.0) * 100.0
        elif pb and pb > 0:
            score = (1.0 - _linear_score(pb, pb_best, pb_worst) / 100.0) * 100.0
        elif dy:
            score = _linear_score(dy, dy_best, dy_worst)
        else:
            score = 50.0

        # Simple weighted combination of available metrics
        return max(0.0, min(100.0, score))

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt-to-equity."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE/Margin: Higher is better
        q_score = 50.0
        if roe is not None:
            # ROE typically ranges from -0.2 to 0.5
            q_score = _linear_score(roe * 100, 25.0, -10.0)
        elif margin is not None:
            q_score = _linear_score(margin * 100, 15.0, -5.0)

        # Debt: Lower is better
        d_score = 50.0
        if debt_eq is not None:
            d_score = (1.0 - _linear_score(debt_eq, 0.5, 2.0)) * 100.0
        
        return (q_score * 0.7) + (d_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG ratio."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # Growth is often linked to ROE in value-investing frameworks
        if roe is not None and roe > 0:
            # Reward higher ROE as a proxy for growth capacity
            g_score = _linear_score(roe * 100, 20.0, 0.0)
        elif peg is not None and peg and peg > 0:
            # Lower PEG is better
            g_score = (1.0 - _linear_score(peg, 0.5, 3.0)) * 100.0
        else:
            g_score = 50.0
            
        return g_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results