"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to reward stability
    "growth": 0.25,    # Increased to reward expansion potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
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
            # Use a small epsilon to prevent zero-score dominance in geometric mean
            # but keep it sensitive enough. (S/100 + 0.01)
            normalized_score = (max(0.0, score_val) / 100.0) + 1e-6
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product_of_powers is already (S1/100)^w1 * (S2/100)^w2 ...
            # To get the geometric mean, we don't need to raise it to (1/total_weight) 
            # if the weights themselves are normalized components of the product.
            # However, to keep it consistent with standard weighted geometric mean:
            # We want Score = (S1^w1 * S2^w2)^ (1 / sum(weights))
            # Since we used normalized scores, we scale it back.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handled separately to ensure valid ranges for log/linear scoring
        score_pe = 0.0
        if pe is not None and pe > 0:
            # Using a wide range for PE (1 to 50)
            score_pe = _log_score(pe, 1.0, 50.0)
            # We want LOW PE to be HIGH score, so invert:
            score_pe = 100.0 - score_pe
        elif pe is not None and pe <= 0:
            score_pe = 100.0 # Negative PE is often a "value" signal (profitable)

        score_pb = 0.0
        if pb is not None and pb > 0:
            score_pb = _log_score(pb, 0.1, 20.0)
            score_pb = 100.0 - score_pb
        elif pb is not None and pb <= 0:
            score_pb = 100.0

        # Average the two components
        return (score_pe + score_pb) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        score_roe = 0.0
        if roe is not None:
            # ROE can be negative, linear interpolation for quality
            score_roe = _linear_score(roe, -0.1, 0.3)
        
        score_debt = 0.0
        if debt_equity is not None:
            # Lower debt is better
            score_debt = _linear_score(debt_equity, 0.0, 1.5)
            score_debt = 100.0 - score_debt

        return (score_roe + score_debt) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        score_roe = 0.0
        if roe is not None:
            score_roe = _linear_score(roe, -0.1, 0.3)

        score_peg = 50.0
        if peg is not None and peg > 0:
            # Lower PEG is better (growth relative to value)
            score_peg = _linear_score(peg, 0.5, 3.0)
            score_peg = 100.0 - score_peg

        return (score_roe + score_peg) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results