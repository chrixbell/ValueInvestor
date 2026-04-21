"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to prioritize stable companies (survivorship bias mitigation)
    "growth": 0.25,    # Increased to capture momentum-like growth signals
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
            # We use a small epsilon to prevent 0.0 from zeroing out the whole product
            # in case of a single bad factor, while still allowing it to be very low.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean is calculated as (Π x_i^w_i)^(1/Σw_i)
            # Since we already applied weights in the power, we just need to normalize by total_weight
            # Note: if we use normalized_score = score/100, the result is in [0, 1]
            # We need to adjust: (product_of_powers)^(1/total_weight) would be wrong if weights aren't normalized.
            # Actually, the formula below is correct for weighted geometric mean if weights sum to 1.
            # If they don't, we treat the product as (S1/100)^w1 * ...
            # To get back to 100 scale: (product_of_powers) * 100 is not right.
            # The correct way to scale a weighted geometric mean back to 100:
            # If product = (s1/100)^w1 * (s2/100)^w2, then composite = product^(1/total_weight) * 100
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use PE if available, else PB. Use forward PE if available as it's more predictive.
        pe = result.valuation.pe_forward or result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        if pe is not None and pe > 0:
            # Low PE is good. Clamp at 50 (very high) and 2 (very low).
            return _linear_score(pe, best=2.0, worst=50.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, best=0.5, worst=10.0)
        elif result.valuation.dividend_yield is not None:
            return _linear_score(result.valuation.dividend_yield, best=0.05, worst=0.0)
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE is a primary quality metric
        if roe is not None:
            roe_score = _linear_score(roe, best=0.20, worst=-0.10)
        else:
            roe_score = 50.0

        # Leverage is a risk metric (lower is better)
        if debt_equity is not None and debt_equity > 0:
            debt_score = _linear_score(debt_equity, best=0.1, worst=2.0)
        else:
            debt_score = 50.0

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and margin trends."""
        peg = result.valuation.peg_ratio
        margin = result.financials.net_margin

        # PEG is a great growth-at-reasonable-price metric
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, best=0.5, worst=3.0)
        else:
            peg_score = 50.0

        # Margin stability/size
        if margin is not None:
            margin_score = _linear_score(margin, best=0.20, worst=0.0)
        else:
            margin_score = 50.0

        return (peg_score * 0.6) + (margin_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        
        return sorted_results