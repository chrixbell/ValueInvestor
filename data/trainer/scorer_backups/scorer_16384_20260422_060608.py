"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for value
    "quality": 0.25,   # Increased quality weight to capture stability
    "growth": 0.25,    # Balanced growth component
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

        # Use arithmetic mean for stability in the presence of potential zero scores 
        # from individual factors (which can happen in geometric mean).
        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_score += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or higher dividend yield is better."""
        # Use a mix of PE and PB for valuation. 
        # We use a fallback mechanism to handle None values.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Score for PE (lower is better)
        if pe is not None and pe > 0:
            # Cap PE at 50 for scoring purposes to avoid extreme outliers
            pe_score = _linear_score(max(0.1, pe), 5.0, 50.0)
        else:
            pe_score = 50.0

        # Score for PB (lower is better)
        if pb is not None and pb > 0:
            pb_score = _linear_score(max(0.1, pb), 1.0, 10.0)
        else:
            pb_score = 50.0

        # Score for Dividend Yield (higher is better)
        if div is not None and div > 0:
            div_score = _linear_score(div, 5.0, 10.0)
        else:
            div_score = 50.0

        # Weighting within the value factor
        return (pe_score * 0.4) + (pb_score * 0.4) + (div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE Score (higher is better)
        if roe is not None:
            roe_score = _linear_score(roe, 15.0, 30.0)
        else:
            roe_score = 50.0

        # Margin Score (higher is better)
        if margin is not None:
            margin_score = _linear_score(margin, 10.0, 30.0)
        else:
            margin_score = 50.0

        # Debt Score (lower is better)
        if debt is not None:
            debt_score = _linear_score(debt, 0.0, 100.0)
        else:
            debt_score = 50.0

        return (roe_score * 0.5) + (margin_score * 0.3) + (debt_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG Score (lower is better, but avoid division by zero/negative)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.0)
        else:
            peg_score = 50.0

        # ROE as a proxy for growth potential/efficiency
        if roe is not None:
            roe_growth_score = _linear_score(roe, 10.0, 25.0)
        else:
            roe_growth_score = 50.0

        return (peg_score * 0.6) + (roe_growth_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results