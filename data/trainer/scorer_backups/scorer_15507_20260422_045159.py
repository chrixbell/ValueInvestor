"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for valuation
    "quality": 0.25,   # Increased quality to ensure robust companies
    "growth": 0.25,    # Balanced growth weight
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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

        # Collect scores with positive weights.
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
            # Normalize score to [0, 1] for geometric mean. 
            # We use max(eps, score/100) to prevent a single zero-score from destroying the entire composite.
            # This allows for "good enough" scoring even if one factor is slightly weak.
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores (S^w) results in a value in [0, 1].
            # We scale it back to [0, 100].
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        # Use PEG as a tie-breaker/supplement if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Simple heuristic: check for valid PE/PB
        if pe is not None and pe > 0:
            # Target low PE
            val_pe = _linear_score(1/pe, 1/50, 1/2) # Low PE is good
        else:
            val_pe = 50.0

        if pb is not None and pb > 0:
            val_pb = _linear_score(1/pb, 1/5, 1/0.1) # Low PB is good
        else:
            val_pb = 50.0

        return (val_pe * 0.6) + (val_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity

        q_roe = 0.0
        if roe is not None:
            # ROE-based score (higher better)
            q_roe = _linear_score(roe, 20.0, -10.0)

        q_debt = 0.0
        if debt_to_equity is not None:
            # Lower debt-to-equity is better
            q_debt = _linear_score(1.0 / max(0.01, debt_to_equity), 1/2, 1/0.1)
        else:
            q_debt = 50.0

        return (q_roe * 0.7) + (q_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        g_roe = 0.0
        if roe is not None:
            g_roe = _linear_score(roe, 15.0, -5.0)

        g_peg = 50.0
        if peg is not None and peg > 0:
            # Lower PEG is better for growth-at-reasonable-price
            g_peg = _linear_score(1/peg, 0.5, 3.0)
        elif peg is None:
            g_peg = 50.0

        return (g_roe * 0.5) + (g_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results