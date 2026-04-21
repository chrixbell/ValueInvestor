"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced value weight to prevent low-quality value traps
    "quality": 0.35,   # Increased quality to ensure robust fundamentals
    "growth": 0.25,    # Balanced growth component
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k] 
            total_weight += weight
            # Use a small epsilon to prevent zero-score issues in geometric mean
            normalized_val = max(0.0001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean formula is (product of x_i^w_i)^(1 / sum(w_i))
            # Since we normalized x_i to [0, 1], the result is in [0, 1]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        # Sort descending: highest score is rank 1
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results

    def _value_score(self, result: ScreeningResult) -> float:
        """Value scoring using PE and PB."""
        # Handle potential None values
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Use PE as primary, fallback to PB or PS
        if pe is not None and pe > 0:
            # Lower PE is better. We use a threshold to avoid extreme outliers.
            # A PE of 0-5 is excellent, 30+ is expensive.
            return _linear_score(pe, best=1.0, worst=30.0) if pe < 30 else (0.0 if pe > 100 else _linear_score(pe, best=30.0, worst=100.0))
        elif pb is not None and pb > 0:
            return _linear_score(pb, best=1.0, worst=10.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, best=1.0, worst=10.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Quality scoring using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        score = 50.0
        # ROE component (higher is better)
        if roe is not None:
            roe_score = _linear_score(roe, best=20.0, worst=-10.0)
            score = score * 0.7 + roe_score * 0.3
        
        # Leverage component (lower is better)
        if debt_equity is not None:
            leverage_score = _linear_score(debt_equity, best=0.1, worst=2.0)
            # If debt is very high, it should penalize heavily
            if debt_equity > 3.0: leverage_score = 0.0
            score = score * 0.7 + leverage_score * 0.3
        else:
            # If no debt info, weight ROE more heavily
            score = roe_score if 'roe_score' in locals() else 50.0

        return score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Growth scoring using PEG and Margin."""
        peg = result.valuation.peg_ratio
        margin = result.financials.net_margin

        score = 50.0
        if peg is not None and peg > 0:
            # PEG < 1 is great, PEG > 3 is expensive growth
            peg_score = _linear_score(peg, best=0.5, worst=3.0)
            score = 0.6 * score + 0.4 * peg_score
        elif margin is not None:
            margin_score = _linear_score(margin, best=20.0, worst=0.0)
            score = 0.6 * score + 0.4 * margin_score
        
        return score