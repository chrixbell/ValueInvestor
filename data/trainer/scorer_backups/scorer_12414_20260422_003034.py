"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced weight to avoid over-reliance on low PE
    "quality": 0.30,   # Increased quality to capture stable returns
    "growth": 0.30,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation.

    *best* is the value that maps to 100 and *worst* maps to 0.
    Values beyond the endpoints are clamped.
    """
    if best == worst:
        return 50.0
    # If the user provides a 'best' that is lower than 'worst' (e.g. PE), 
    # the math still holds: (5 - 40) / (1 - 40) = 1.
    # We need to ensure the subtraction order matches the intention.
    # Let's normalize: if best < worst, we swap to ensure (value - worst)/(best - worst) works.
    # Wait, the current logic: if value=5, best=1, worst=40 -> (5-40)/(1-40) = -35/-39 = 0.89
    # If value=40, best=1, worst=40 -> (40-40)/(1-40) = 0.
    # If value=1, best=1, worst=40 -> (1-40)/(1-40) = 1.
    # This works perfectly for any input!
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

        # Momentum is a placeholder — set to 50 (neutral).
        momentum_score = 50.0 

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process)
        
        # Geometric mean approach: (S1^w1 * S2^w2 ...)^(1/sum(wi))
        # We use max(0, score) to ensure we don't have issues with negative numbers.
        # However, since our scores are clamped [0, 100], we just divide by 100.
        product_of_powers = 1.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            # Normalize score to [0, 1] range for geometric mean calculation.
            normalized_val = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        
        scores = []
        # Use PE if available and positive
        if pe is not None and pe > 0:
            scores.append(_linear_score(pe, 1.0, 40.0))
        # Use PB if available and positive
        if pb is not None and pb > 0:
            scores.append(_linear_score(pb, 0.5, 12.0))
        # Use PS if available and positive
        if ps is not None and ps > 0:
            scores.append(_linear_score(ps, 0.5, 10.0))
            
        if not scores:
            return 50.0
        # Return average of available value metrics
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        net_margin = result.financials.net_margin

        scores = []
        if roe is not None:
            # High ROE is better. 30% is great, -10% is bad.
            scores.append(_linear_score(roe, 30.0, -10.0))
        if debt_to_equity is not None:
            # Low leverage is better. 0 is best, 2.0 is high.
            scores.append(_linear_score(debt_to_equity, 0.0, 2.5))
        if net_margin is not None:
            # High margin is better. 20% is great, -5% is bad.
            scores.append(_linear_score(net_margin, 20.0, -5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        scores = []
        if peg is not None and peg > 0:
            # Low PEG is a growth/value hybrid. 0.5 best, 3.0 worst.
            scores.append(_linear_score(peg, 0.5, 3.0))
        if roe is not None:
            # High ROE often correlates with growth. 25% best, -5% worst.
            scores.append(_linear_score(roe, 25.0, -5.0))

        if not scores:
            return 50.0
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