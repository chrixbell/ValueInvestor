"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture stable earnings power
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # but keep the zero-penalty logic intact.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product calculation above already effectively handles the weighting.
            # (S1^w1 * S2^w2) is already the weighted product.
            # We just need to ensure it's not zeroed out by a single 0 weight if we want to be robust.
            # But per current logic:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield implies better value."""
        # Using a combination of PE and PB for valuation. 
        # If PE is None, we fall back to PB or dividend yield.
        
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Base score components
        v1 = 0.0
        if pe is not None and pe > 0:
            # Target PE range 5-20. Higher than 40 is poor, lower than 5 is "value trap" risk but still good value.
            v1 = _linear_score(pe, 5.0, 40.0)
            v1 = 100.0 - v1 # Invert so lower PE is better
        elif pb is not None and pb > 0:
            v1 = _linear_score(pb, 0.5, 5.0)
            v1 = 100.0 - v1

        # Dividend yield component (higher is better)
        v2 = 0.0
        if dy is not None:
            v2 = _linear_score(dy, 0.01, 0.10) # 1% to 10%

        # Weighted average of value components
        if pe is not None:
            return (v1 * 0.7) + (v2 * 0.3)
        return v2 if dy is not None else v1

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and margin, lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.0
        
        # ROE score: Target high ROE (e.g., 5% to 30%)
        # We use a simple linear scale for ROE.
        roe_score = _linear_score(roe * 100, 5.0, 30.0) if roe is not None else 50.0
        # Margin score: Target high net margin (e.g., 5% to 25%)
        margin_score = _linear_score(margin * 100, 5.0, 25.0) if margin is not None else 50.0
        # Leverage score: Lower debt to equity is better (e.g., 0 to 1)
        debt_score = _linear_score(debt, 0.0, 1.5) if debt is not None else 50.0
        debt_score = 100.0 - debt_score

        # Combine quality factors
        return (roe_score * 0.4) + (margin_score * 0.3) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio

        # Growth is often captured by ROE in static snapshots
        roe_score = _linear_score(roe * 100, 5.0, 30.0) if roe is not None else 50.0
        
        # PEG score: lower is better (target 0.5 to 2.0)
        peg_score = 50.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.5)
            peg_score = 100.0 - peg_score

        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results