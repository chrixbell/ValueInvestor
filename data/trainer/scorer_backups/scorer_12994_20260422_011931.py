"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more breathing room for quality/growth
    "quality": 0.25,   # Increased to capture stable earnings and solvency
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Use a small epsilon to prevent zero-score annihilation in geometric mean
            # while still allowing low scores to pull down the composite.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculates the value score based on PE and PB ratios."""
        # Factors where lower is better (PE, PB)
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # We use a simple combination of PE and PB to represent valuation.
        # If both are missing, we return a neutral 50.
        if pe is None and pb is None:
            return 50.0
            
        # Use PE as primary, PB as secondary if available.
        # In Chinese markets, PE is often more relevant for growth-oriented value.
        if pe is not None and pe > 0:
            # A reasonable PE range for value scoring [1, 30]
            return _linear_score(pe, 2.0, 30.0) if pe > 0 else 0.0
        elif pb is not None and pb > 0:
            return _linear_score(pb, 0.5, 5.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates the quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # Quality is a mix of profitability (ROE) and solvency (Debt/Equity).
        # If data is missing, we attempt to score based on available fields.
        score_roe = 50.0
        score_debt = 50.0

        if roe is not None:
            # Higher ROE is better. Map 0-30% range to 0-100 score.
            score_roe = _linear_score(roe * 100, 5.0, 25.0)
        
        if debt_equity is not None:
            # Lower Debt/Equity is better. Map 0-2 range to 100-0 score.
            # We invert the linear score logic: best is 0, worst is 2.
            score_debt = _linear_score(0.0, 0.0, debt_equity) if debt_equity > 0 else 100.0
        elif debt_equity is None and roe is not None:
            # If we only have ROE, prioritize it.
            score_debt = 50.0

        # Combine: 60% ROE, 40% Debt/Equity
        return (score_roe * 0.6) + (score_debt * 0.4)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates the growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # Growth scoring: PEG is a key metric for growth-at-reasonable-price.
        if peg is not None and peg > 0:
            # Lower PEG is better.
            return _linear_score(peg, 0.5, 2.0)
        elif roe is not None:
            # If PEG is missing, use ROE as a proxy for growth potential.
            return _linear_score(roe * 100, 5.0, 30.0)
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks all results and assigns ranks based on composite_score."""
        for r in results:
            self.score(r)

        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)

        for i, res in enumerate(sorted_results):
            res.rank = i + 1

        return sorted_results