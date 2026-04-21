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
    "quality": 0.25,   # Increased quality weight to ensure fundamental stability
    "growth": 0.25,    # Balanced growth weight
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
            # We use a small epsilon to prevent zero-product issues in geometric mean
            # but allow the score to be low if sub-scores are low.
            normalized_score = max(0.0, score_val) / 100.0
            # To prevent a single zero from wiping out everything, we allow a tiny floor 
            # if the weight is significant, but here we keep it pure to penalize bad stocks.
            product_of_powers *= (normalized_score) ** weight

        if total_weight > 0:
            # The geometric mean of normalized scores [0, 1]
            # Since we are multiplying (S/100)^w, the result is already in [0, 1]
            # We don't need to raise it to (1/total_weight) if the weights are normalized to sum to 1.
            # However, to handle non-normalized weights:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        pv = result.valuation.ps_ratio

        scores = []
        # PE factor (lower is better, but avoid negative/zero issues)
        if pe is not None and pe > 0:
            # Clamp PE to a reasonable range for scoring [1, 50]
            scores.append(_linear_score(1.0/pe, 1.0/50.0, 1.0/1.0))
        elif pe is not None and pe <= 0:
            scores.append(100.0) # Highly profitable or negative PE is often good for value
        
        if pb is not None and pb > 0:
            scores.append(_linear_score(1.0/pb, 1.0/10.0, 1.0/0.5))
        
        if pv is not None and pv > 0:
            scores.append(_linear_score(1.0/pv, 1.0/5.0, 1.0/0.5))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        scores = []
        # ROE score (higher is better)
        scores.append(_linear_score(roe, 0.30, -0.10))
        # Margin score (higher is better)
        scores.append(_linear_score(margin, 0.20, -0.10))
        # Leverage score (lower is better)
        if debt_equity > 0:
            scores.append(_linear_score(1.0/debt_equity, 1.0/2.0, 1.0/0.5))
        else:
            scores.append(100.0)

        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        scores = []
        if peg is not None and peg > 0:
            scores.append(_linear_score(1.0/peg, 1.0/3.0, 1.0/0.5))
        else:
            # If PEG is not available, we rely on ROE or give neutral score
            scores.append(50.0)
        
        # Incorporate ROE as a proxy for internal growth capability
        scores.append(_linear_score(roe, 0.20, -0.10))

        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results