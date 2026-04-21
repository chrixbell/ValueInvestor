"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Stabilizing value weight to prevent over-concentration in low-quality cheap stocks.
    "quality": 0.25,   # Increased quality weight to filter for better-performing companies.
    "growth": 0.25,    # Balanced growth weight to capture upside potential.
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score.
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

        # Add a small epsilon to prevent zero-score annihilation in geometric mean
        epsilon = 1e-6

        total_weight = 0.0
        product_of_powers = 1.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use (score + epsilon) / 100 to ensure non-zero terms for geometric mean
            product_of_powers *= ((score_val + epsilon) / 100.0) ** weight

        if total_weight > 0:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use PEG-like logic or simple ratio comparison
        # We use a mixture of PE and PB. 
        # If one is missing, we lean on the other.
        if pe is not None and pe > 0:
            # Typical reasonable PE range for screening: 1 to 40
            pe_score = _linear_score(pe, 2.0, 40.0)
        else:
            pe_score = 50.0

        if pb is not None and pb > 0:
            # Typical reasonable PB range: 0.5 to 10
            pb_score = _linear_score(pb, 0.5, 10.0)
        else:
            pb_score = 50.0

        # Invert linear score because lower is better
        return (pe_score + pb_score) / 2.0 if pe is not None and pb is not None else (pe_score if pe is not None else pb_score)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity

        q_score = 50.0
        if roe is not None:
            # ROE maps to 100 at 20%, 0 at -5%
            roe_score = _linear_score(roe, 20.0, -5.0)
            q_score = roe_score
        
        if debt_eq is not None:
            # Debt-to-equity: higher is worse. 
            # Map 0 debt to 100, 2.0 (200%) debt to 0.
            debt_score = _linear_score(2.0, 0.0, debt_eq)
            if roe is None:
                q_score = debt_score
            else:
                # Combine with ROE (weighting quality components)
                q_score = (roe_score + debt_score) / 2.0 if 'roe_score' in locals() else debt_score

        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, 0.5 is great, 3.0 is bad
            peg_score = _linear_score(peg, 0.5, 3.0)
            return peg_score
        elif roe is not None:
            # Fallback to ROE if PEG is unavailable
            return _linear_score(roe, 15.0, -5.0)
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results