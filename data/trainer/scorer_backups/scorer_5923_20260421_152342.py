"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow quality/growth more breathing room
    "quality": 0.25,   # Increased to emphasize stability
    "growth": 0.25,    # Increased to capture upside-potential stocks
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
            # Use a small epsilon to prevent zero-score issues in geometric mean while allowing penalty
            safe_score = max(0.001, score_val) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # The geometric mean formula with weights is (Π S_i^w_i)^(1 / Σ w_i)
            # Since we already applied weights in the loop, we just need to handle the scaling.
            # However, if we use (S/100)^weight, the result is in [0, 1].
            # To get it back to [0, 100], we multiply by 100.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PEG as a tie-breaker or component if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Fallback logic to ensure we always have a value
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # We want to score low PE/PB as high
        # Using a simple linear interpolation with reasonable bounds for A-shares/HK
        v_pe = _linear_score(pe, 5.0, 30.0)
        v_pb = _linear_score(pb, 1.0, 5.0)
        
        # If PEG is available and low, it's a strong value signal
        if peg is not None and peg > 0:
            v_peg = _linear_score(peg, 0.5, 2.0)
            return (v_pe * 0.4 + v_pb * 0.3 + v_peg * 0.3)
        
        return (v_pe * 0.6 + v_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # Normalize: higher ROE/Margin (target ~20%), lower Debt (target ~0.5)
        q_roe = _linear_score(roe * 100, 5.0, 30.0)
        q_margin = _linear_score(margin * 100, 5.0, 25.0)
        q_debt = _linear_score(debt, 0.0, 1.5) # Invert: higher debt -> lower score
        # Since _linear_score is (val-worst)/(best-worst), and for debt, 'best' should be 0.
        # Let's manually handle debt:
        q_debt = max(0.0, min(100.0, (1.5 - debt) / 1.5 * 100.0)) if debt < 1.5 else 0.0
        if debt >= 1.5: q_debt = 0.0
        elif debt <= 0: q_debt = 100.0

        return (q_roe * 0.4 + q_margin * 0.3 + q_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and growth potential."""
        # Use ROE as a proxy for growth-quality if revenue growth isn't explicit
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        
        # Growth score: High ROE and low PEG is a classic value-growth combination
        g_roe = _linear_score(roe * 100, 5.0, 25.0)
        g_peg = _linear_score(peg, 0.5, 2.0) if peg > 0 else 0.0
        
        return (g_roe * 0.6 + g_peg * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results