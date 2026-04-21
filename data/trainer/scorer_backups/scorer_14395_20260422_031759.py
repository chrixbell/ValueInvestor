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
    "quality": 0.25,   # Increased to prioritize companies with strong balance sheets
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
            # Use a small epsilon to prevent math domain errors if score is 0
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product_of_powers is (S1/100)^w1 * (S2/100)^w2 ...
            # To get the result back to 0-100, we don't need a power of (1/total_weight)
            # unless the weights were not normalized to 1.0. 
            # However, since we want a geometric mean-like structure:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or higher dividend yield is better."""
        # Use a mix of PE and PB for valuation
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Handle potential None or negative values (e.g., negative PE)
        # For simplicity in this iteration, we use a weighted linear combination of normalized components
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Map PE 1-50 to 100-0
            pe_score = _linear_score(pe, 5.0, 40.0)
        elif pe is not None and pe <= 0:
            pe_score = 100.0 # Negative PE is often a sign of loss, but in simple models it's "cheap"

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1.0, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 100.0

        div_score = 0.0
        if div is not None:
            div_score = _linear_score(div, 2.0, 8.0)

        return (pe_score * 0.4 + pb_score * 0.4 + div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE score: assume typical range -10% to 30%
        roe_score = _linear_score(roe, 0.15, -0.1)
        # Margin score: assume typical range 0% to 25%
        margin_score = _linear_score(margin, 0.20, 0.0)
        # Debt score: lower is better
        debt_score = _linear_score(debt, 0.5, 2.0)

        return (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0

        # Growth component: ROE
        roe_score = _linear_score(roe, 0.20, -0.05)
        
        # PEG component: lower is better (growth at reasonable price)
        peg_score = 0.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.0)
        elif peg is not None and peg <= 0:
            peg_score = 100.0

        return (roe_score * 0.5 + peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results