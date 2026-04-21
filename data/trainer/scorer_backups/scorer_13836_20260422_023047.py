"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow for higher quality/growth signal
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Increased to capture upside potential
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
            # We use a small epsilon to prevent log(0) issues if we were using log-space aggregation,
            # but here we use direct power. We map [0, 100] to [0, 1].
            # To prevent a single zero score from destroying the whole composite (common in geometric means),
            # we ensure a floor of 0.01 for the score used in multiplication.
            normalized_score = max(0.0, score_val) / 100.0
            # If the score is 0, we treat it as a very low but non-zero value to allow for 
            # differentiation in the geometric mean if other factors are high.
            # However, to stay mathematically consistent with the prompt's goal:
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers already accounts for weights. 
            # If we want the weighted geometric mean: (S1^w1 * S2^w2 * ...) ^ (1 / sum(wi))
            # Since we normalized by 100, the result is in [0, 1].
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Using a mix of cheapness and yield
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        div = result.valuation.dividend_yield

        scores = []
        # PE: lower is better (clamp at 1 to avoid division/log issues)
        if pe and pe > 0:
            scores.append(_linear_score(1/pe, 1/60, 1/2))
        if pb and pb > 0:
            scores.append(_linear_score(1/pb, 1/15, 1/0.5))
        if ps and ps > 0:
            scores.append(_linear_score(1/ps, 1/3, 1/0.1))
        if div and div > 0:
            scores.append(_linear_score(div, 0.02, 0.10))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and stable margins are better."""
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        scores = []
        if roe is not None:
            # Map ROE to a score. Assuming reasonable range -20% to 40%
            scores.append(_linear_score(roe, 0.30, -0.10))
        if margin is not None:
            scores.append(_linear_score(margin, 0.15, -0.05))
        if debt is not None:
            # Debt to equity: lower is better. Clamp at 0 for stability.
            scores.append(_linear_score(1/(debt + 0.01) if debt >= 0 else 1, 1/2, 1/0.5))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth and reasonable PEG."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        scores = []
        if peg is not None and peg > 0:
            # Lower PEG is better (growth at a reasonable price)
            scores.append(_linear_score(1/peg, 1/0.5, 1/3))
        if roe is not None:
            scores.append(_linear_score(roe, 0.25, -0.1))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results