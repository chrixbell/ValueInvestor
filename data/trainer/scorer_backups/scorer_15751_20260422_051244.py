"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced value weight slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased quality to reward stability and reduce risk
    "growth": 0.25,    # Balanced growth with quality
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
            # but allow the zero to propagate via scaling.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers already applies the weights. 
            # To get a weighted geometric mean: (S1^w1 * S2^w2 ...)^(1/sum_w)
            # We use the product of powers directly as it is already normalized by weights if we consider 
            # that S_normalized = (S/100).
            # However, to ensure it returns to [0, 100] correctly:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB."""
        # Use PE or PB as primary value metrics
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback logic for None values
        pe = pe if pe is not None and pe > 0 else 25.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Score based on low PE/PB
        # We use a simplified approach: combine pe and pb if available.
        v_score = _linear_score(1.0, 1.0, 30.0) # Placeholder-like structure
        # Re-implementing logic to be more robust:
        if pe > 0 and pb > 0:
            # If PE is low, higher score. If PB is low, higher score.
            # We'll use a simple heuristic: 1/PE and 1/PB.
            # But since we have the _linear_score, let's use it with reasonable bounds.
            pe_s = _linear_score(pe, 5.0, 40.0)
            pb_s = _linear_score(pb, 1.0, 5.0)
            return (pe_s + pb_s) / 2.0
        elif pe:
            return _linear_score(pe, 5.0, 40.0)
        else:
            return _linear_score(pb, 1.0, 5.0)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        de = result.financials.debt_to_equity
        
        # Handle None: default to neutral/average values
        roe_val = roe if roe is not None else 0.10 # 10%
        de_val = de if de is not None else 0.5    # 50%

        # Higher ROE is better
        roe_s = _linear_score(roe_val * 100, 5.0, 30.0)
        # Lower Debt/Equity is better (inverse linear)
        de_s = _linear_score(1.0 / (de_val + 0.01), 0.1, 2.0)
        
        return (roe_s + de_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # Handle None: default to neutral/average values
        peg_val = peg if peg is not None and peg > 0 else 1.5
        roe_val = roe if roe is not None else 0.10

        # Lower PEG is better
        peg_s = _linear_score(1.0 / peg_val, 0.5, 3.0)
        # Higher ROE is better (growth component)
        roe_s = _linear_score(roe_val * 100, 5.0, 30.0)

        return (peg_s + roe_s) / 2.0