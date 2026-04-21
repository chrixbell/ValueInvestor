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
    "quality": 0.25,   # Increased to penalize low-quality companies more effectively
    "growth": 0.25,    # Balanced with quality
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

        # Momentum is a placeholder
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

        # We use a modified geometric mean approach. 
        # To prevent a single zero score from destroying the entire composite, 
        # we use an epsilon to allow low but non-zero scores to propagate.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] range. Use epsilon to avoid log(0) issues in math logic
            normalized_score = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores scaled back to 100
            # Since weights are already applied in the power, we don't divide by total_weight 
            # unless weights are not normalized to sum to 1.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use a combination of PE and PB for valuation
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback to 20 if PE is None or negative (to avoid issues)
        pe = pe if (pe and pe > 0) else 20.0
        pb = pb if (pb and pb > 0) else 2.0

        # Score for PE (lower is better)
        pe_score = _linear_score(pe, best=5.0, worst=40.0)
        # Score for PB (lower is better)
        pb_score = _linear_score(pb, best=1.0, worst=10.0)

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        debt_to_equity = result.financials.debt_to_equity if (result.financials.debt_to_equity is not None) else 0.5
        
        # ROE: Higher is better (using linear score with reasonable bounds)
        roe_score = _linear_score(roe, best=0.20, worst=-0.10)
        # Debt/Equity: Lower is better (clamped to avoid negative scores or extreme sensitivity)
        # 0% debt = 100, 100% debt = 0
        de_score = _linear_score(debt_to_equity, best=0.0, worst=1.5)
        
        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: Screening_result if False else ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        # In a real scenario, we'd use revenue growth. Here we use ROE as a proxy for internal growth capacity
        # and PEG (if available) to balance valuation with growth.
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        peg = result.valuation.peg_ratio if (result.valuation.peg_ratio and result.valuation.peg_ratio > 0) else 2.0
        
        # ROE is a strong indicator of growth capability
        roe_score = _linear_score(roe, best=0.25, worst=-0.05)
        # PEG: Lower is better (attractive growth-to-value ratio)
        peg_score = _linear_score(peg, best=0.5, worst=3.0)
        
        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results