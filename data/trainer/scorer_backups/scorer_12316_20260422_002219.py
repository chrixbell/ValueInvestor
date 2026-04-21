"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Slightly reduced to allow more room for quality/growth
    "quality": 0.30,   # Increased to penalize low-quality companies that might look cheap
    "growth": 0.25,    # Balanced growth component
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

        # To prevent a single zero sub-score from destroying the entire composite score
        # (which happens in geometric means), we use a small epsilon for the base.
        epsilon = 1e-6

        total_weight = 0.0
        product_of_powers = 1.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use (score/100 + epsilon) to ensure we don't take log of zero or multiply by zero
            normalized_score = (score_val / 100.0) + epsilon
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula with weights is (Π x_i^w_i)^(1 / Σ w_i)
            # If we use the epsilon-adjusted normalized score, we scale back to 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use PE if available, else PB. If both are missing, return neutral 50.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Define reasonable bounds for the Chinese market/A-shares
        pe_best, pe_worst = 15.0, 60.0
        pb_best, pb_worst = 1.5, 8.0

        if pe is not None and pe > 0:
            # We want lower PE to be better.
            # Since _linear_score assumes higher value = better, we invert the logic
            # by passing worst as best and vice versa, or simply handling it.
            # Let's use a logic: score = 100 at low PE, 0 at high PE.
            # We will use a helper to map: higher value is better for the scoring function.
            # For PE, "lower" is better. So we map pe_worst to 100 and pe_best to 0.
            return _linear_score(pe_worst, pe_best, pe_worst) if pe_worst != pe_best else 50.0
        elif pb is not None and pb > 0:
            return _linear_score(pb_worst, pb_best, pb_worst) if pb_worst != pb_best else 50.0
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # Quality components: High ROE is good, Low Debt-to-Equity is good.
        # We combine them into a single score.
        roe_score = 0.0
        if roe is not None:
            # ROE range: -20% to 40%
            roe_score = _linear_score(roe, 30.0, -20.0)
        
        debt_score = 50.0
        if debt_equity is not None:
            # Debt/Equity range: 0 to 2.0
            debt_score = _linear_score(debt_equity, 0.2, 2.5)
        
        # If ROE is available, it's a much stronger quality signal.
        if roe is not None:
            return (roe_score * 0.7) + (debt_score * 0.3)
        return debt_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # Growth is often captured by the relationship between growth and valuation
        if roe is not None and peg is not None and peg > 0:
            # High ROE + Low PEG = Goldmine. 
            # We use a combination of growth potential and value-for-growth.
            # Mapping: High ROE is good, Low PEG is good.
            roe_part = _linear_score(roe, 25.0, -10.0)
            peg_part = _linear_score(peg, 0.5, 3.0)
            return (roe_part * 0.5) + (peg_part * 0.5)
        elif roe is not None:
            return _linear_score(roe, 20.0, -10.0)
        elif peg is not None and peg > 0:
            return _linear_score(peg, 0.5, 3.0)
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        
        return sorted_results