"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow for more quality/growth balance
    "quality": 0.25,   # Increased weight to capture more stability
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
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

        # Momentum is a placeholder
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
            # Use a small epsilon to prevent zeroing out the whole score if one factor is 0
            # but still allow low scores to penalize heavily.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores (scaled back to 100)
            # result = (prod(s_i^w_i))^(1/sum(w_i)) * 100
            # However, in the previous version, product_of_powers was already (s1^w1 * s2^w2...)
            # To get the correct geometric mean: (s1^w1 * s2^w2...) ^ (1/total_weight) is NOT the standard 
            # weighted geometric mean. The correct formula is (s1^(w1/W) * s2^(w2/W)...).
            # Since we are using weights directly in the power, we actually just need to 
            # calculate: product_of_powers = (s1/100)^w1 * (s2/100)^w2 ...
            # Then the result is product_of_powers^(1/total_weight) * 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher Dividend Yield."""
        # We use a combination of PE, PB, and Dividend Yield.
        pe = result.valuation.pe_ratio if result.valuation.pe_ratio is not None and result.valuation.pe_ratio > 0 else None
        pb = result.valuation.pb_ratio if result.valuation.pb_ratio is not None and result.valuation.pb_ratio > 0 else None
        div = result.valuation.dividend_yield if result.valuation.dividend_yield is not None else None

        scores = []
        if pe:
            # PE range 0-50 is common for value. Clamp to avoid extreme outliers.
            scores.append(_linear_score(min(pe, 50), 50, 1))
        if pb:
            scores.append(_linear_score(min(pb, 15), 15, 0.1))
        if div is not None:
            # High dividend yield is good for value.
            scores.append(_linear_score(min(div, 10), 10, 0))
        
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/Margin and lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else None
        margin = result.financials.net_margin if result.financials.net_margin is not None else None
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else None
        
        scores = []
        if roe is not None:
            # ROE can be negative, but for scoring we treat high positive as best.
            # Map -20% to 50% range to [0, 100]
            scores.append(_linear_score(roe * 100, 50, -20))
        if margin is not None:
            scores.append(_linear_score(margin * 100, 20, -10))
        if debt is not None:
            # Lower debt = higher score. 
            # Map 0-200% range to [100, 0]
            scores.append(_linear_score(debt * 100, 200, 0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else None
        roe = result.financials.roe if result.financials.roe is not None else None
        
        scores = []
        if peg is not None:
            # PEG < 1 is great. High PEG is bad.
            scores.append(_linear_score(peg, 1.5, 3))
        if roe is not None:
            # ROE as a proxy for growth potential in this context.
            scores.append(_linear_score(roe * 100, 30, 0))
            
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            r.score(result=r) # This ensures scores are calculated if not already
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results