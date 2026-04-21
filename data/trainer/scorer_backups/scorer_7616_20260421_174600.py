"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Slightly reduced to balance with quality/growth
    "quality": 0.25,   # Increased weight to emphasize stability
    "growth": 0.30,    # Increased weight to capture upside potential
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
            # Use a small epsilon to prevent zero-score annihilation in geometric mean
            # while still allowing low scores to pull the composite down.
            safe_score = max(0.001, score_val) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # We use a mix of forward PE and PB. 
        # Lower is better for value.
        pe = result.valuation.pe_forward or result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle negative/None PE (common in loss-making companies)
        if pe is None or pe <= 0:
            pe_score = 30.0 # Neutral/low score for negative earnings
        else:
            # Map PE to score: lower is better. 
            # Using a range of 0 to 40 for PE (where 40 is 'worst' in this context)
            pe_score = _linear_score(pe, best=5.0, worst=40.0)

        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Map PB to score: lower is better.
            pb_score = _linear_score(pb, best=1.0, worst=6.0)

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe or 0.0
        debt_equity = result.financials.debt_to_equity or 0.0

        # ROE is a primary quality metric.
        # Map ROE (e.g., 0% to 30%) to high score.
        roe_score = _linear_score(roe * 100, best=20.0, worst=-5.0)

        # Debt/Equity: Lower is better for quality.
        if debt_equity <= 0:
            de_score = 100.0
        else:
            de_score = _linear_score(debt_equity, best=0.2, worst=2.0)

        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe or 0.0

        # PEG: Lower is better (growth at reasonable price)
        if peg is None or peg <= 0:
            peg_score = 50.0
        else:
            peg_score = _linear_score(peg, best=0.5, worst=3.0)

        # ROE also acts as a proxy for growth/efficiency
        roe_score = _linear_score(roe * 100, best=25.0, worst=-10.0)

        return (peg_score * 0.5) + (roe_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results