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
    "quality": 0.25,   # Increased to prioritize fundamental stability
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent zero-score annihilation in geometric mean
            # but allow for low scores.
            norm_score = max(0.001, score_val / 100.0)
            product_of_powers *= (norm_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        # Sort descending by composite_score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher Dividend Yield are better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Use pe_forward if available, else pe_ratio
        pe_val = result.valuation.pe_forward if result.valuation.pe_forward is not None else pe
        
        # If PE is negative or None, we assign a baseline low score for value
        if pe_val is None or pe_val <= 0:
            pe_s = 20.0
        elif pe_val < 5:
            pe_s = 100.0
        elif pe_val < 15:
            pe_s = 80.0
        elif pe_val < 30:
            pe_s = 50.0
        else:
            pe_s = 20.0

        pb_s = 50.0
        if pb is not None:
            if pb < 1.5: pb_s = 100.0
            elif pb < 3: pb_s = 70.0
            elif pb < 6: pb_s = 40.0
            else: pb_s = 10.0

        dy_s = 50.0
        if dy is not None:
            dy_s = min(100.0, dy * 200.0) # 5% yield = 100 score

        # Combine using a simple weighted approach for the sub-score
        return (pe_s * 0.4) + (pb_s * 0.4) + (dy_s * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin and lower leverage."""
        roe = result.financials.roe or 0.0
        margin = result.financials.net_margin or 0.0
        debt = result.financials.debt_to_equity or 0.0
        assets = result.financials.total_assets or 1.0

        # ROE Score (Targeting high ROE)
        roe_s = _linear_score(roe, 25.0, 0.0)
        
        # Margin Score (Targeting high margin)
        margin_s = _linear_score(margin, 20.0, 0.0)

        # Leverage Score (Lower is better)
        if debt <= 0:
            lev_s = 100.0
        else:
            lev_s = _linear_score(1/debt, 1.0, 5.0) # Higher score for lower debt

        return (roe_s * 0.5) + (margin_s * 0.3) + (lev_s * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe or 0.0
        peg = result.valuation.peg_ratio

        # PEG is a growth-at-reasonable-price metric
        if peg is None or peg <= 0:
            # If PEG is unavailable/negative, rely on ROE
            peg_s = 50.0
        else:
            # PEG < 1 is great, PEG > 2 is poor
            peg_s = _linear_score(1.0/peg, 2.0, 0.5)

        # ROE is a proxy for growth quality
        roe_s = _linear_score(roe, 20.0, -5.0)

        return (peg_s * 0.6) + (roe_s * 0.4)