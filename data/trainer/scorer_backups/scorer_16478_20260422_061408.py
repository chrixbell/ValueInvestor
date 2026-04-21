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
    "quality": 0.25,   # Increased to capture more stable fundamentals
    "growth": 0.25,    # Increased to capture expansionary potential
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
            # We use a small epsilon (1e-6) to prevent math domain errors with 0.0
            # but allow the zero score to propagate through the geometric mean.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores (S/100) is calculated.
            # If any score is 0, the product becomes 0.
            # The result is scaled back to [0, 100].
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # We use a combination of PE and PB to capture different valuation aspects.
        # If values are None, we fallback to a neutral score or other metrics.
        v1 = 0.0
        if pe is not None and pe > 0:
            # Using a wide range for PE (1 to 50)
            v1 = _linear_score(pe, 2.0, 40.0)
            # Invert: lower PE is better
            v1 = 100.0 - v1
        elif pe is not None and pe <= 0:
            v1 = 100.0 # Negative PE is often a strong value signal in certain contexts

        v2 = 0.0
        if pb is not None and pb > 0:
            v2 = _linear_score(pb, 0.5, 10.0)
            v2 = 100.0 - v2
        elif pb is not None and pb <= 0:
            v2 = 100.0

        # Combine using simple average if both exist
        if pe is not None and pb is not None:
            return (v1 + v2) / 2.0
        return v1 if pe is not None else v2

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        q1 = 50.0
        if roe is not None:
            # ROE-based score (higher is better)
            q1 = _linear_score(roe, 5.0, 30.0)
        
        q2 = 50.0
        if debt_equity is not None:
            # Low debt-to-equity is better
            q2 = _linear_score(debt_equity, 0.0, 1.5)
            q2 = 100.0 - q2
        
        if roe is not None and debt_equity is not None:
            return (q1 + q2) / 2.0
        return q1 if roe is not None else q2

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        g1 = 50.0
        if peg is not None and peg > 0:
            # Lower PEG is better (growth adjusted value)
            g1 = _linear_score(peg, 0.5, 3.0)
            g1 = 100.0 - g1
        elif peg is not None and peg <= 0:
            g1 = 100.0 # Negative PEG implies growth is higher than PE (or negative PE)

        g2 = 50.0
        if roe is not None:
            # High ROE as a proxy for growth efficiency
            g2 = _linear_score(roe, 5.0, 30.0)

        if peg is not None and roe is not None:
            return (g1 + g2) / 2.0
        return g1 if peg is not None else g2

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite_score."""
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            # Rank is 1-based. Handle ties if necessary, but standard sort is fine for ρ.
            res.rank = i + 1
        return results