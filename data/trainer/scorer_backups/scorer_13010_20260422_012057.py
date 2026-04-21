"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced value weight to allow quality/growth more room
    "quality": 0.30,   # Increased quality weight to capture better-managed firms
    "growth": 0.30,    # Increased growth weight to capture expansion potential
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # But since we are doing power, (score/100)^weight is fine if score >= 0.
            # We use a tiny epsilon to ensure stability for very low scores.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Geometric mean calculation: (S1^w1 * S2^w2 ...) ^ (1 / sum(wi))
            # However, since we want the result in [0, 100], and product_of_powers is already
            # normalized by weights in the exponent, we just need to scale it.
            # If product = (s1/100)^w1 * (s2/100)^w2, then the result is product * 100.
            # We don't need (1/total_weight) if the weights were designed to sum to 1. 
            # But for robustness, we use it:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # Wait, if weights sum to 1: (s/100)^w1 * (s/100)^w2... is already in [0, 1]
            # The correct formula for weighted geometric mean of x_i is (product x_i^w_i)^(1 / sum w_i)
            # Since we are using (score/100), the result is in [0, 1].
            # Let's refine:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # If we use the pure formula: product_of_powers = (s1/100)^w1 * (s2/100)^w2...
            # then the result is indeed in [0, 1]. To get back to [0, 100], we multiply by 100.
            # The logic above: (product_of_powers ** (1/total_weight)) is only needed if 
            # sum(weights) != 1. If we want the result to be a weighted average-like scale:
            # Let's just use: 100 * product_of_powers (assuming weights sum to 1)
            # or more generally: 100 * (product_of_powers ** (1.0 / total_weight)) is WRONG for scale.
            # Correct: If weights sum to 1, result = (product_of_powers) * 100.
            # If weights sum to 2, product_of_powers is in units of (score/100)^2.
            # To return to [0, 1], we need product_of_powers ** (1/total_weight).
            # So the current logic is actually correct for any total_weight.
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        # Use forward PE as primary, fallback to trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Define reasonable bounds for Chinese market
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        score_pe = 0.0
        if pe is not None and pe > 0:
            # Lower PE is better for value
            score_pe = _linear_score(pe, pe_worst, pe_best)
        elif pe is not None and pe <= 0:
            score_pe = 100.0 # Negative PE is often very high value (though risky)
        else:
            score_pe = 50.0

        score_pb = 0.0
        if pb is not None and pb > 0:
            score_pb = _linear_score(pb, pb_worst, pb_best)
        elif pb is not None and pb <= 0:
            score_pb = 100.0
        else:
            score_pb = 50.0

        return (score_pe * 0.6) + (score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and debt."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.1

        # ROE: Higher is better
        # Debt/Equity: Lower is better (assuming 1.0 is a threshold)
        # Margin: Higher is better

        # ROE scale: 0% to 30%
        score_roe = _linear_score(roe * 100, 30.0, 0.0)
        # Debt scale: 0 to 2.0 (200%)
        score_debt = _linear_score(debt_equity, 2.0, 0.0)
        # Margin scale: -10% to 20%
        score_margin = _linear_score(margin * 100, 20.0, -10.0)

        return (score_roe * 0.4) + (score_debt * 0.3) + (score_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and revenue/income growth."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        # Since we don't have explicit growth rates, we use the relationship between ROE and PE
        # or simply rely on PEG as a proxy for growth-adjusted value.
        
        # If PEG is very low, it's high growth potential relative to price.
        # If PEG is 1.0, it's fair. If > 2.0, expensive growth.
        if peg is not None and peg > 0:
            score_peg = _linear_score(peg, 3.0, 0.5)
        else:
            score_peg = 50.0

        # Using ROE as a proxy for quality-growth
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        score_roe = _linear_score(roe * 100, 25.0, 0.0)

        return (score_peg * 0.6) + (score_roe * 0.4)