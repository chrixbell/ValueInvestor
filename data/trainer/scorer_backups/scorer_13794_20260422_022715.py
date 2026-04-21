"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Focus on value capture
    "quality": 0.25,   # Quality as a stabilizer
    "growth": 0.25,    # Growth for alpha
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k] 
            total_weight += weight
            # Use a small epsilon to avoid math errors with zero scores in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            # If score is 0, we treat it as a very small epsilon to allow the product to exist
            # but still reflect the penalty.
            if normalized_score == 0 and weight > 0:
                normalized_score = 1e-6

            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Weighted geometric mean calculation
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score using PE and PB ratios."""
        # Use PE as primary, PB as secondary. 
        # High PE is bad for value (unless growth is huge).
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        if pe is not None and pe > 0:
            # Standardize PE: lower is better. Range [0.5, 50]
            pe_score = _linear_score(pe, best=1.0, worst=50.0)
        else:
            # Fallback to PB if PE is missing or negative
            if pb is not None and pb > 0:
                pe_score = _linear_score(pb, best=0.5, worst=10.0)
            else:
                pe_score = 50.0

        # Combine with Dividend Yield if available
        dy = result.valuation.dividend_yield
        if dy is not None and dy > 0:
            dy_score = _linear_score(dy, best=8.0, worst=0.0)
            return (pe_score * 0.7) + (dy_score * 0.3)
        
        return pe_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity

        # ROE is usually the strongest quality signal
        if roe is not None:
            roe_score = _linear_score(roe, best=25.0, worst=-5.0)
        else:
            roe_score = 50.0

        if debt_to_equity is not None:
            # Lower debt is better. Clamp at 200% (2.0) as worst.
            debt_score = _linear_score(debt_to_equity, best=0.0, worst=2.0)
        else:
            debt_score = 50.0

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: Screening_Result) -> float:
        """Computes growth score using PEG ratio and Revenue growth logic."""
        # In this dataset, we use ROE/Growth proxies via PEG if available.
        peg = result.valuation.peg_ratio
        
        if peg is not None and peg > 0:
            # Lower PEG is better (Growth at a reasonable price)
            growth_score = _linear_score(peg, best=0.5, worst=3.0)
        else:
            # If PEG is unavailable, use ROE as a proxy for internal growth potential
            roe = result.financials.roe
            if roe is not None:
                growth_score = _linear_score(roe, best=20.0, worst=-10.0)
            else:
                growth_score = 50.0
        
        return growth_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results