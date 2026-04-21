"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced value weight to make room for quality/growth
    "quality": 0.25,   # Increased quality weight to capture fundamental stability
    "growth": 0.30,    # Increased growth weight for potential alpha in A-shares
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
            # Use a small epsilon to prevent 0.0 from zeroing out the whole product in geometric mean
            # but allow it to be a strong penalty.
            normalized_score = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # To avoid math domain error with 0.0, we handle the base case
            if product_of_powers == 0:
                composite_score = 0.0
            else:
                # If using geometric mean, the power is applied to the product.
                # The formula (S1^w1 * S2^w2)^ (1/sum_w) is equivalent to product of (Si^(wi/sum_w))
                # We simplify by just using the product directly if weights are already relative.
                # But to keep it consistent with previous logic:
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        # Valuation metrics - lower is better (PE, PB)
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        
        # Use PE as primary, fallback to PB or PS
        if pe is not None and pe > 0:
            # Typical healthy PE range for value focus: 5 to 30
            return _linear_score(pe, 5.0, 30.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 1.0, 5.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, 1.0, 5.0)
        else:
            return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        # Quality metrics - higher is better (ROE, Margin), lower is better (Debt)
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        q_score = 0.0
        count = 0

        if roe is not None:
            # ROE range -10% to 30%
            q_score += _linear_score(roe * 100, -10.0, 30.0)
            count += 1
        if margin is not None:
            # Margin range -5% to 20%
            q_score += _linear_score(margin * 100, -5.0, 20.0)
            count += 1
        if debt is not None:
            # Debt to equity range 0 to 2.0 (200%)
            # Higher debt is worse, so we map it inversely: 2.0 -> 0, 0 -> 100
            # But _linear_score(value, best, worst) where best is 0 and worst is 2.
            # To use _linear_score(value, best=0, worst=2) would give 0 for value=2.
            # Let's use a simplified approach:
            d_val = max(0.0, min(2.0, debt))
            q_score += (1.0 - (d_val / 2.0)) * 100.0
            count += 1

        return q_score / count if count > 0 else 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        # Growth/Efficiency - higher is better (ROE, PEG)
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        g_score = 0.0
        count = 0

        if peg is not None and peg > 0:
            # Low PEG is better. Range 0.5 to 2.5
            g_score += _linear_score(peg, 0.5, 2.5)
            count += 1
        if roe is not None:
            # Higher ROE as a growth proxy
            g_score += _linear_score(roe * 100, 5.0, 25.0)
            count += 1

        return g_score / count if count > 0 else 50.0