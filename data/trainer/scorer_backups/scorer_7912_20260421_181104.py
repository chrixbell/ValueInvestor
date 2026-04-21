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
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Increased to capture expansion potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst:float) -> float:
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
            # Use a small epsilon to prevent log(0) issues if score is 0, 
            # though geometric mean naturally handles 0.
            normalized_score = score_val / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of (S/100)^w is already the weighted geometric mean logic.
            # If we want to return a score in [0, 100], we take the product.
            # Since sum(weights) might not be 1, we normalize by total_weight.
            # (S1^w1 * S2^w2)^(1/total_weight)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use PEG as a secondary value check if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        peg = result.valuation.peg_ratio

        # We prioritize PE and PB, but use PEG to penalize high-growth value traps
        # Target ranges for linear scoring: PE [0, 30], PB [0, 5], PS [0, 5]
        # If PE is negative, it's treated as the worst (0) for this calculation
        pe_val = pe if pe is not None and pe > 0 else 999.0
        pb_val = pb if pb is not None and pb > 0 else 999.0
        ps_val = ps if ps is not None and ps > 0 else 999.0

        # Simple heuristic: average of normalized scores
        # We want low values to give high scores. 
        # Let's define "worst" as a very high PE/PB and "best" as 0.5
        def inverse_linear(val, best, worst):
            if val <= best: return 100.0
            if val >= worst: return 0.0
            return (worst - val) / (worst - best) * 100.0

        s_pe = inverse_linear(pe_val, 1.0, 30.0)
        s_pb = inverse_linear(pb_val, 1.0, 6.0)
        s_ps = inverse_linear(ps_val, 1.0, 5.0)

        # Combine
        return (s_pe * 0.4 + s_pb * 0.3 + s_ps * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and Margin, lower Debt to Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE and Margin: Higher is better. We use sigmoid-like clamping.
        # Assuming ROE/Margin are in decimal (0.15 = 15%)
        # Target ROE: 20% (best), 0% (worst)
        # Target Margin: 15% (best), 0% (worst)
        s_roe = max(0.0, min(100.0, (roe * 100 + 20) / 20 * 100 if roe is not None else 0.0)) # overly simplified
        # Let's use a more robust approach:
        
        def scale_up(val, best, worst):
            if val >= best: return 100.0
            if val <= worst: return 0.0
            return (val - worst) / (best - worst) * 100.0

        # ROE/Margin: 25% is great, -5% is bad
        s_roe = scale_up(roe if roe is not None else -0.1, 0.25, -0.1)
        s_margin = scale_up(margin if margin is not None else -0.1, 0.20, -0.1)
        # Debt: 0 is best, 1.5 (150%) is worst
        s_debt = max(0.0, min(100.0, (1.5 - (debt if debt is not None else 1.5)) / 1.5 * 100.0))

        return (s_roe * 0.4 + s_margin * 0.4 + s_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE/Growth, lower PEG."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe if result.financials.roe is not None else 0.1
        
        # PEG: lower is better (but avoid negative/zero issues)
        # A PEG of 1.0 is neutral, 0.5 is great, 3.0 is bad.
        if peg is not None and peg > 0:
            s_peg = max(0.0, min(100.0, (3.0 - peg) / (3.0 - 0.1) * 100.0)) if peg < 3.0 else 0.0
        else:
            s_peg = 50.0

        # ROE as a proxy for growth/efficiency
        s_roe = max(0.0, min(100.0, (roe * 100 + 10) / 30 * 100 if roe is not None else 0.0))

        return (s_peg * 0.6 + s_roe * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results