"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to reward stability
    "growth": 0.25,    # Increased to capture upside potential
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
                # Add a small epsilon to avoid zero in geometric mean if one factor is 0
                weighted_scores_to_process[k] = max(score_val, 0.01)

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        total_weight = sum(self.weights[k] for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            # Normalize score to [0, 1] for geometric mean calculation
            product_of_powers *= (score_val / 100.0) ** weight

        # The geometric mean is (product_of_powers)^(1/total_weight)
        # But since we used weights directly in the exponent, product_of_powers is already 
        # effectively (S1^w1 * S2^w2...). We just need to normalize the exponent.
        # To keep it in [0, 100], we use: (product_of_powers)^(1/total_weight) * 100
        # However, if weights are normalized to sum to 1, product_of_powers is the result.
        # To be safe for any weight sum:
        result.composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        
        # In case of floating point precision issues resulting in slightly > 100
        result.composite_score = min(100.0, result.composite_score)

        return result

    def _value_score(self, result: ScreeningResult) -> float:
        # Combine PE and PB for a balanced valuation score. 
        # Using PEG as a secondary check to ensure growth is priced reasonably.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Handle negative PE (loss making)
        if pe is None or pe <= 0:
            pe_score = 20.0 # Low baseline for loss makers
        else:
            # Goal: Lower PE is better. Clamp at 50 for "worst" and 5 for "best".
            pe_score = _linear_score(pe, best=5.0, worst=50.0)

        if pb is None or pb <= 0:
            pb_score = 20.0
        else:
            # Goal: Lower PB is better. Clamp at 1 for "best" and 10 for "worst".
            pb_score = _linear_score(pb, best=1.0, worst=10.0)

        # PEG check: If growth is expensive, penalize
        peg_score = 50.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, best=0.5, worst=3.0)

        return (pe_score * 0.4) + (pb_score * 0.4) + (peg_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.0
        
        # ROE is primary quality metric (Target: 20% best, -10% worst)
        roe_score = _linear_score(roe * 100, best=20.0, worst=-10.0)
        # Margin (Target: 15% best, 0% worst)
        margin_score = _linear_score(margin * 100, best=15.0, worst=0.0)
        # Debt (Target: 0 best, 100 worst) -> We want low debt
        debt_score = _linear_score(debt, best=0.0, worst=1.5)

        return (roe_score * 0.5) + (margin_score * 0.3) + (debt_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        # Growth is often correlated with ROE, but we want to capture the "growth" aspect.
        # Since specific growth rates aren't provided, we use ROE and Revenue/Net Income as proxies.
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        rev_growth = 0.0 # Placeholder if not available, but we can use ROE/Margin interaction
        
        # Using a non-linear approach: High ROE + Positive Net Income = Growth potential
        net_inc = result.financials.net_income if result.financials.net_income is not None else 0.0
        
        # Score based on ROE (high is better) and Net Income presence
        roe_score = _linear_score(roe * 100, best=25.0, worst=-5.0)
        
        # Interaction: If net income is positive and growing (relative to assets), higher score
        assets = result.financials.total_assets if result.financials.total_assets else 1.0
        roa = (net_inc / assets) * 100 if assets > 0 else 0
        roa_score = _linear_score(roa, best=10.0, worst=-5.0)

        return (roe_score * 0.6) + (roa_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results