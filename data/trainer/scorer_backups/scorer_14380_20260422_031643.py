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
    "quality": 0.25,   # Increased to prioritize stable companies
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
            # Using a small epsilon to prevent zero causing issues in geometric mean 
            # while still allowing low scores to pull the composite down.
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean calculation: (Product of score^weight) ^ (1/total_weight)
            # Since we use normalized scores [0, 1], the result stays in [0, 1].
            # We scale back to [0, 100].
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE as primary, fallback to PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle potential None or zero values
        pe = pe if (pe is not None and pe > 0) else 20.0
        pb = pb if (pb is not None and pb > 0) else 1.0

        # Scoring: Lower PE/PB is better
        # We use a wide range for thresholds to capture various market conditions
        score_pe = _linear_score(1.0, 25.0, 60.0) # Low PE is good
        score_pb = _linear_score(1.0, 4.0, 15.0)  # Low PB is good
        
        return (score_pe * 0.6) + (score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if (result.financials.debt_to_equity is not None and result.financials.debt_to_equity > 0) else 1.0
        
        # Higher ROE is better
        score_roe = _linear_score(roe, 20.0, 5.0) # 20% ROE -> 100, 5% ROE -> 0
        # Lower Debt/Equity is better
        score_debt = _linear_score(1.0/debt_equity, 0.5, 3.0)
        
        return (score_roe * 0.7) + (score_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Revenue growth (via ROE/Profitability proxy)."""
        peg = result.valuation.peg_ratio if (result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0) else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # Lower PEG is better (growth at a reasonable price)
        score_peg = _linear_score(1.0, 1.5, 4.0)
        # Higher ROE as a proxy for growth quality
        score_growth = _linear_score(roe, 15.0, 2.0)
        
        return (score_peg * 0.5) + (score_growth * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results