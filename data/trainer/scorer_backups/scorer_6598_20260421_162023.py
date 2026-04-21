"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more influence from quality/growth
    "quality": 0.25,   # Increased to emphasize stability and profitability
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
            # Use a small epsilon to prevent zero-score from zeroing out the entire product 
            # if it's just a single bad factor in a geometric mean, but keep the penalty.
            # We use (score_val / 100.0) as the base.
            normalized_val = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Geometric mean: (prod(S_i^w_i)) ^ (1/sum(w_i))
            # Since we are using normalized scores, the result is in [0, 1]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use forward PE if available as it's more relevant for 6-month returns
        pe_val = result.valuation.pe_forward if pe is not None and result.valuation.pe_forward is not None else pe
        
        # If PE/PB are negative or None, we handle them as worst-case for value
        # but to avoid math errors in log_score, we clamp to a small positive or use linear
        # For simplicity and robustness, let's use weighted linear scores for value.
        
        pe_score = 0.0
        if pe_val is not None and pe_val > 0:
            # Assume PE range [1, 40] for scoring. Low is better.
            pe_score = _linear_score(pe_val, 1.0, 40.0)
        elif pe_val is not None and pe_val <= 0:
            # Negative PE (profitable) is actually good for value.
            pe_score = 100.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.1, 10.0)
        elif pb is not None and pb <= 0:
            pb_score = 100.0

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE component (Higher is better)
        roe_score = 0.0
        if roe is not None:
            # Map ROE (-20% to 40%) to [0, 100]
            roe_score = _linear_score(roe, 40.0, -20.0)
        
        # Leverage component (Lower is better)
        leverage_score = 0.0
        if debt_equity is not None:
            # Map Debt/Equity (0 to 2.0) to [100, 0]
            leverage_score = _linear_score(debt_equity, 0.0, 2.0)
        elif debt_equity is None:
            leverage_score = 50.0

        return (roe_score * 0.7) + (leverage_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio
        margin = result.financials.gross_margin

        # PEG component (Lower is better, but only if positive)
        peg_score = 0.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            # Negative PEG implies high growth relative to PE; highly desirable.
            peg_score = 100.0

        # Margin component (Higher is better)
        margin_score = 0.0
        if margin is not None:
            # Map Margin (-10% to 50%) to [0, 100]
            margin_score = _linear_score(margin, 50.0, -10.0)
        elif margin is None:
            margin_score = 50.0

        return (peg_score * 0.5) + (margin_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results