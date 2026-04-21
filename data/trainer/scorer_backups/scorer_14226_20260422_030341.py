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
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent zero-score issues in geometric mean
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Re-scale back to [0, 100]
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Combines PE and PB ratios for value scoring."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        scores = []
        # PE is better when lower
        if pe is not None and pe > 0:
            # Clamping PE between 1 and 50 for scoring purposes
            clamped_pe = max(1.0, min(50.0, pe))
            scores.append(_linear_score(clamped_pe, 1.0, 50.0))
        
        if pb is not None and pb > 0:
            clamped_pb = max(0.1, min(20.0, pb))
            scores.append(_linear_score(clamped_pb, 0.5, 20.0))
            
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Combines ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        scores = []
        if roe is not None:
            # ROE can be negative; map typical ranges. 
            # Assuming -20% to 40% range for scoring.
            scores.append(_linear_score(roe * 100, 40.0, -20.0))
            
        if debt_equity is not None:
            # Lower debt-to-equity is better
            # Scale input to handle percentage or decimal
            de_val = debt_equity * 100 if debt_equity > 2.0 else debt_equity
            if de_val <= 0: de_val = 1.0 # Avoid division/log issues
            scores.append(_linear_score(de_val, 0.5, 100.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Combines ROE and PEG ratio."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        scores = []
        if roe is not None:
            scores.append(_linear_score(roe * 100, 30.0, -10.0))
            
        if peg is not None and peg > 0:
            # Lower PEG is better (growth at a reasonable price)
            clamped_peg = max(0.1, min(5.0, peg))
            scores.append(_linear_score(clamped_peg, 0.5, 5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results