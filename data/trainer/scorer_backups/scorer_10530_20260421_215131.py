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
    "quality": 0.25,   # Increased to reward stable companies
    "growth": 0.25,    # Increased to capture upside potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
    if value <= 0 or best <= 0 or worst <= 0:
        return 0.0

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0
        
        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


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

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0,
        }

        # Filter for weights > 0 to avoid math errors in geometric mean
        active_weights = {k: v for k, v in self.weights.items() if v > 0}
        
        if not active_weights:
            result.composite_score = 50.0
            return result

        # We use a weighted arithmetic mean of normalized scores for stability.
        # Geometric mean can be extremely sensitive to a single zero sub-score.
        total_weight = sum(active_weights.values())
        weighted_sum = 0.0
        
        for k, weight in active_weights.items():
            # Use a small epsilon to avoid zero-multiplication issues in some logic, 
            # but here we just take the score directly.
            score_val = scores.get(k, 50.0)
            weighted_sum += (score_val / 100.0) * weight

        # Final score is weighted average scaled back to 100
        result.composite_score = (weighted_sum / total_weight) * 100.0
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or higher Dividend Yield is better."""
        # Use a mix of PE and PB if available, fallback to dividend yield
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Scoring logic: Low PE/PB is good.
        # We define 'best' as 0 (or near 0) and 'worst' as a high threshold.
        if pe is not None and pe > 0:
            # PE-based score (lower is better)
            return _linear_score(1/pe, 1/5.0, 1/50.0) if pe > 0 else 0.0
        elif pb is not None and pb > 0:
            return _linear_score(1/pb, 1/2.0, 1/10.0)
        elif dy is not None and dy > 0:
            return _linear_score(dy, 0.05, 0.0)
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt to equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        score = 50.0
        # ROE component (Higher is better)
        if roe is not None:
            roe_score = _linear_score(roe, 0.20, -0.10)
            score = 0.6 * roe_score + 40.0 # Scale to avoid total zero
        
        # Debt component (Lower is better)
        if debt_equity is not None:
            # Mapping higher debt to lower score
            debt_score = _linear_score(1.0/max(0.01, debt_equity), 1.0/0.1, 1.0/2.0)
            score = 0.4 * debt_score + (0.6 * score if roe is not None else 0)
            # If ROE was none, we just use debt_score as base
            if roe is None: score = debt_score

        # Ensure we handle the case where both are None
        if roe is None and debt_equity is None:
            return 50.0
        
        # Re-adjusting to keep it within [0, 100]
        return max(0.0, min(100.0, score))

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        if roe is not None and peg is not None and peg > 0:
            # Growth quality: High ROE with low PEG
            return _linear_score(roe / peg, 0.3, -0.1)
        elif roe is not None:
            return _linear_score(roe, 0.20, -0.1)
        elif peg is not None and peg > 0:
            return _linear_score(1/peg, 1.0, 0.1)
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results