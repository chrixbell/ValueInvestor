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
    "quality": 0.25,   # Increased to reward stable business models
    "growth": 0.25,    # Balanced with quality for a robust fundamental profile
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
            # Use a small epsilon to prevent zero-out if one factor is 0
            # but keep the penalty for bad companies.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The power (1/total_weight) is actually redundant here because the 
            # weights are already applied inside the product. We just scale back to 100.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use forward PE if available, else trailing. 
        # If PE is negative or None, we use PB as a fallback/secondary measure.
        pe_val = result.valuation.pe_forward if (pe is not None and pe > 0) else pe
        
        score = 0.0
        if pe_val is not None and pe_val > 0:
            # Higher PE = lower score. Using a reasonable cap for 'worst' PE.
            score = _linear_score(1/pe_val, 1/2.0, 1/50.0)
        elif pb is not None and pb > 0:
            score = _linear_score(1/pb, 1/0.5, 1/10.0)
        else:
            # If no valuation data, return a neutral-low score
            score = 20.0
        return score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe
        d_e = result.financials.debt_to_equity

        # ROE component (higher is better)
        roe_score = 0.0
        if roe is not None:
            # Normalize ROE (assuming reasonable range -20% to 40%)
            roe_score = _linear_score(roe, 0.30, -0.10)
        else:
            roe_score = 50.0

        # Leverage component (lower is better)
        leverage_score = 0.0
        if d_e is not None:
            # Normalize D/E (assuming 0 to 2.0 range)
            leverage_score = _linear_score(1/(d_e + 0.1), 1/0.1, 1/3.0)
        else:
            leverage_score = 50.0

        return (roe_score + leverage_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # ROE component (higher is better)
        roe_score = 0.0
        if roe is not None:
            roe_score = _linear_score(roe, 0.20, -0.10)
        else:
            roe_score = 50.0

        # PEG component (lower is better)
        peg_score = 0.0
        if peg is not None and peg > 0:
            peg_score = _linear_score(1/peg, 1/0.5, 1/3.0)
        elif peg is None:
            peg_score = 50.0
        else:
            peg_score = 0.0

        return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results