"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced to allow more room for quality/growth
    "quality": 0.30,   # Increased to capture more stable fundamental drivers
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
            # Using a small epsilon to prevent zero-score annihilation in geometric mean
            normalized_val = max(0.01, score_val) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of powers is already (S1/100)^w1 * (S2/100)^w2...
            # To get the geometric mean, we don't actually need to raise it to (1/total_weight) 
            # if we want the result to stay in [0, 100] range relative to the weights.
            # However, if the weights sum to 1, product_of_powers is already the geometric mean.
            # We scale by 100 to return to [0, 100].
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Use forward PE if available, otherwise trailing
        target_pe = pe if pe is not None else (result.valuation.pe_forward if result.valuation.pe_forward is not None else None)
        
        # Score components
        s_pe = 0.0
        if target_pe is not None and target_pe > 0:
            # Lower PE is better. Mapping range [1 to 40] -> [100 to 0]
            s_pe = _linear_score(target_pe, 1.0, 40.0)
        elif target_pe is not None and target_pe <= 0:
            s_pe = 100.0 # Negative PE is often a sign of high growth or turnaround
        else:
            s_pe = 50.0

        s_pb = 0.0
        if pb is not None and pb > 0:
            s_pb = _linear_score(pb, 0.5, 10.0)
        elif pb is not None and pb <= 0:
            s_pb = 100.0
        else:
            s_pb = 50.0

        s_ps = 0.0
        if ps is not None and ps > 0:
            s_ps = _linear_score(ps, 0.5, 10.0)
        elif ps is not None and ps <= 0:
            s_ps = 100.0
        else:
            s_ps = 50.0

        return (s_pe * 0.4 + s_pb * 0.3 + s_ps * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE Score (Higher is better)
        if roe is not None:
            s_roe = _linear_score(roe, 15.0, -5.0)
        else:
            s_roe = 50.0

        # Debt/Equity Score (Lower is better)
        if debt_equity is not None:
            # Avoid division by zero or negative if input is weird
            s_debt = _linear_score(debt_equity, 0.1, 2.0)
        else:
            s_debt = 50.0

        # Margin Score (Higher is better)
        if margin is not None:
            s_margin = _linear_score(margin, 15.0, -5.0)
        else:
            s_margin = 50.0

        return (s_roe * 0.4 + s_debt * 0.3 + s_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG Score (Lower is better)
        if peg is not None and peg > 0:
            s_peg = _linear_score(peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            s_peg = 100.0
        else:
            s_peg = 50.0

        # ROE as a proxy for growth potential
        if roe is not None:
            s_roe = _linear_score(roe, 20.0, -5.0)
        else:
            s_roe = 50.0

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