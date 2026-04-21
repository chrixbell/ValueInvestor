"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced value weight to allow more room for quality/growth
    "quality": 0.30,   # Increased quality weight to penalize bad balance sheets
    "growth": 0.25,    # Growth weight maintained for potential upside
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

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
        }

        # We use a weighted sum approach for better stability in rank-based correlation 
        # when using different factor distributions.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_sum += score_val * weight
                total_weight += weight

        if total_weight > 0:
            # Normalize the sum back to [0, 100] range
            composite_score = weighted_sum / (total_weight / 100.0) if total_weight != 0 else 50.0
            # Safety clamp
            composite_score = max(0.0, min(100.0, composite_score))
        else:
            composite_score = 50.0

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Using a mix of valuation metrics to avoid single-metric bias.
        # We prioritize PE and PB but include PS for context.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Define reasonable bounds for Chinese A-shares/HK
        # If PE is negative, it's handled by being much worse than a low positive PE.
        v1 = _linear_score(pe if pe is not None and pe > 0 else 999, 5.0, 40.0) if pe is not None else 0.0
        v2 = _linear_score(pb if pb is not None and pb > 0 else 999, 1.0, 5.0) if pb is not None else 0.0
        v3 = _linear_score(ps if ps is not None and ps > 0 else 999, 1.0, 10.0) if ps is not None else 0.0

        # Average the available valuation scores
        valid_scores = [v for v in [v1, v2, v3] if v > 0 or (pe is not None or pb is not None or ps is not None)]
        if not valid_scores:
            return 50.0
        return sum(valid_scores) / len(valid_scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE score (Targeting 15-20% as best)
        q_roe = _linear_score(roe if roe is not None else 0, 15.0, 30.0) if roe is not None else 0.0
        # Debt score (Lower debt is better)
        q_debt = _linear_score(debt if debt is not None and debt >= 0 else 100, 0.5, 2.0) if debt is not None else 0.0
        # Margin score
        q_margin = _linear_score(margin if margin is not None else 0, 10.0, 25.0) if margin is not None else 0.0

        # Quality is often a combination of profitability and solvency
        scores = []
        if roe is not None: scores.append(q_roe)
        if debt is not None: scores.append(100 - q_debt) # Invert because lower debt is better
        if margin is not None: scores.append(q_margin)

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a key growth-at-reasonable-price metric
        g_peg = _linear_score(peg if peg is not None and peg > 0 else 50, 0.5, 2.0) if peg is not None else 0.0
        # Growth-related ROE (using a higher threshold for growth)
        g_roe = _linear_score(roe if roe is not None else 0, 20.0, 40.0) if roe is not None else 0.0

        scores = []
        if peg is not None: scores.append(g_peg)
        if roe is not None: scores.append(g_roe)

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort all results by composite score and assign ranks."""
        # Sort descending (higher score is better)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results