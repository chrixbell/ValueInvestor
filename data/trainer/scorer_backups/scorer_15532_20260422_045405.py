"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Slightly reduced to allow more room for quality/growth
    "quality": 0.30,   # Increased to prioritize robust balance sheets and profitability
    "growth": 0.25,    # Maintains growth component to capture expansion potential
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
            # We add a small epsilon (1e-6) to prevent math errors if score_val is 0
            # but keep the zero-penalty logic intact.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean scaled back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Using a mix of valuation metrics to avoid single-metric outliers.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Filter for valid positive ratios
        valid_pe = pe is not None and pe > 0.1
        valid_pb = pb is not None and pb > 0.1
        valid_ps = ps is not None and ps > 0.1

        if valid_pe and valid_pb:
            # Combine PE and PB as a proxy for valuation
            return _linear_score(1.0 / (pe/pb if pe != 0 else 1), 2.0, 0.5) # Heuristic
        elif valid_pe:
            return _linear_score(1.0/pe, 20.0, 1.0)
        elif valid_pb:
            return _linear_score(1.0/pb, 5.0, 0.5)
        elif valid_ps:
            return _linear_score(1.0/ps, 5.0, 0.5)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # Quality score combines profitability (ROE/Margin) and solvency
        # We use a scale where higher is better. 
        # ROE is often normalized around 15-20% in healthy companies.
        roe_score = _linear_score(roe, 0.30, -0.10)
        # Debt-to-equity: lower is better. Clamp to avoid negative scores.
        debt_score = _linear_score(1.0 / (max(0.1, debt_equity)), 2.0, 0.2)
        margin_score = _linear_score(margin, 0.20, -0.10)

        return (roe_score * 0.5 + debt_score * 0.3 + margin_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG is a powerful growth/value hybrid metric.
        if peg is not None and peg > 0:
            # A PEG of 1.0 is neutral, < 1 is good, > 2 is bad.
            peg_score = _linear_score(1.0/peg, 2.0, 0.5)
        else:
            # Fallback to ROE if PEG is unavailable
            peg_score = _linear_score(roe, 0.30, -0.10)

        return peg_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results