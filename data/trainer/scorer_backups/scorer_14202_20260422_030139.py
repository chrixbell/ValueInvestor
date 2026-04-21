"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced slightly to allow more balance with quality/growth
    "quality": 0.30,   # Increased to focus on fundamental stability
    "growth": 0.25,    # Maintained growth component
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
            # Using a small epsilon to avoid zero issues with geometric mean
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean logic is built into the product of powers:
            # (S1^w1 * S2^w2) is the product. To normalize it back to a 0-1 range,
            # we don't need the (1/total_weight) exponent if we want to maintain 
            # the scale, because (S/100)^weight already handles scaling.
            # However, to keep it robust:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        # Use PE if available, else PB
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle negative PE (common in loss-making stocks)
        # We want to avoid extreme values but capture low PE.
        # If PE is negative, we treat it as a very high (bad) value or use PB.
        
        if pe is not None and pe > 0:
            # A low PE is better. Range 0-30 for mapping.
            return _linear_score(pe, best=1.0, worst=30.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, best=1.0, worst=10.0)
        elif pe is not None and pe <= 0:
            # If PE is negative, it's usually a bad "value" signal for this specific metric
            return 5.0 
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # Quality is a mix of high ROE and low leverage
        roe_score = 50.0
        if roe is not None:
            # ROE can be negative, but we map a reasonable range. 
            # High ROE is good. 0% to 30% range.
            roe_score = _linear_score(roe * 100, best=30.0, worst=-10.0)
        
        debt_score = 50.0
        if debt_equity is not None:
            # Lower debt/equity is better. 0 to 1.5 range.
            debt_score = _linear_score(debt_equity, best=0.1, worst=2.0)

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # Growth score: Low PEG is good, high ROE is a proxy for growth potential
        peg_score = 50.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, best=0.5, worst=3.0)
        elif peg is None:
            peg_score = 50.0
        else: # Negative PEG (growth > earnings)
            peg_score = 80.0

        roe_score = 50.0
        if roe is not None:
            # Higher ROE often correlates with growth-ready companies
            roe_score = _linear_score(roe * 100, best=25.0, worst=-5.0)

        return (peg_score * 0.6) + (roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results