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
    "growth": 0.25,    # Increased to capture expansionary potential
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
            # Using a small epsilon to prevent math domain errors with 0.0 scores in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # If the product is 0, we want to avoid issues. We scale back by total_weight weight logic
            # Since (S^w) is already calculated, we don't need to apply 1/total_weight exponent again
            # unless the weights were not normalized. We assume they are roughly 1.0 total.
            # To keep it robust: (S1^w1 * S2^w2) where sum(w)=1.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort results by composite score and assign ranks."""
        # Sort descending (higher score is better)
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield maps to better score."""
        # We use a mix of valuation metrics. 
        # Pe/Pb are often non-linear, but we use linear interpolation for stability.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Defaulting to neutral-ish values if None
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        div = div if div is not None else 0.0

        # Scoring: Low PE is good, low PB is good, high dividend is good
        # We use a simple additive model for value sub-components
        v1 = _linear_score(pe, 40.0, 5.0)  # PE: 5 is best (100), 40 is worst (0)
        v2 = _linear_score(pb, 3.0, 0.5)   # PB: 0.5 is best (100), 3.0 is worst (0)
        v3 = _linear_score(div, 5.0, 0.0)  # Div: 5% is best (100), 0% is worst (0)

        return (v1 * 0.4 + v2 * 0.4 + v3 * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE/Margin: higher is better (clamped)
        q_roe = max(0.0, min(100.0, (roe + 0.2) * 100.0 / 0.5)) # Scale ROE (0.2 is neutral)
        q_margin = max(0.0, min(100.0, (margin + 0.1) * 100.0 / 0.3)) # Scale Margin
        # Debt: lower is better (0.5 is neutral)
        q_debt = _linear_score(debt, 0.2, 1.0)

        return (q_roe * 0.4 + q_margin * 0.4 + q_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG → higher score."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.1
        
        # PEG: Lower is better (1.0 is neutral)
        g_peg = _linear_score(peg, 0.5, 3.0)
        # ROE as a proxy for growth potential/efficiency
        g_roe = max(0.0, min(100.0, (roe + 0.1) * 100.0 / 0.3))

        return (g_peg * 0.5 + g_roe * 0.5)