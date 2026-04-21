"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow for more quality/growth balance
    "quality": 0.25,   # Increased to penalize bad balance sheets more effectively
    "growth": 0.25,    # Balanced with quality
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

        # Use a small epsilon to prevent zero-division/log issues in geometric mean
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Scale to [0, 1] and ensure it's slightly above 0 for stability
            normalized_score = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers is already (S1^w1 * S2^w2...). 
            # To get the weighted geometric mean, we take the (1/total_weight) power.
            # However, since our weights are normalized to sum to 1 (or close to it),
            # we apply the power to normalize the scale correctly.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield improve score."""
        v = result.valuation
        # Using a mix of PE and PB for value, with dividend yield as a safety net.
        # We use linear scoring for simplicity in this iteration.
        pe = v.pe_ratio if v.pe_ratio is not None and v.pe_ratio > 0 else 30.0
        pb = v.pb_ratio if v.pe_ratio is not None and v.pb_ratio is not None and v.pb_ratio > 0 else 5.0
        div = v.dividend_yield if v.dividend_yield is not None and v.dividend_yield > 0 else 0.0
        
        # Score components: lower PE/PB is better, higher dividend is better.
        pe_score = _linear_score(1/pe if pe != 0 else 1, 1/30.0, 1/5.0) # Placeholder logic
        # More robust: map PE to score directly
        pe_score = _linear_score(1.0/pe, 1.0/40.0, 1.0/5.0) # Lower PE -> higher score
        pb_score = _linear_score(1.0/pb, 1.0/10.0, 1.0/1.0)
        
        # Use PEG to refine value if available
        peg = v.peg_ratio if v.peg_ratio is not None and v.peg_ratio > 0 else 2.0
        
        # Simplified: Combine PE/PB logic
        score = (pe_score + pb_score) / 2.0
        return score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin and lower leverage."""
        f = result.financials
        if not f:
            return 50.0

        roe = f.roe if f.roe is not None else 0.0
        margin = f.net_margin if f.net_margin is not None else 0.0
        debt = f.debt_to_equity if f.debt_to_equity is not None else 0.5
        
        # ROE and Margin (higher is better)
        # We use a scale: 0-50% ROE is common for top stocks
        roe_score = _linear_score(roe, 0.4, 0.1) # 40% is best, 10% is worst
        margin_score = _linear_score(margin, 0.2, 0.05)
        # Debt (lower is better): mapped so high debt = low score
        debt_score = _linear_score(1.0 / (1.0 + debt), 1.0/(1.0+2.0), 1.0/(1.0+0.5))
        
        return (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        v = result.valuation
        f = result.financials
        if not f or not v:
            return 50.0

        roe = f.roe if f.roe is not None else 0.0
        peg = v.peg_ratio if v.peg_ratio is not None and v.peg_ratio > 0 else 5.0
        
        # Growth is often correlated with ROE in value models
        roe_score = _linear_score(roe, 0.3, 0.1)
        # PEG: lower is better (growth at a reasonable price)
        peg_score = _linear_score(1.0/peg, 1.0/5.0, 1.0/0.5)
        
        return (roe_score * 0.5 + peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results