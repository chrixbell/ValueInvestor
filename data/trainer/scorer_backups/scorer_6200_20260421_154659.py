"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Lowered slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased to reward stable, high-margin companies
    "growth": 0.25,    # Increased to capture expansion potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best: float, worst: float) -> float:
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
            # Use a small epsilon to prevent zero-out if one factor is 0
            # but keep it close to the geometric mean logic.
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle case where PE is None or non-positive
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Map PE between 1 and 30 to 100-0
            pe_score = _linear_score(pe, 1.0, 30.0)
        
        pb_score = 0.0
        if pb is not None and pb > 0:
            # Map PB between 0.5 and 10 to 100-0
            pb_score = _linear_score(pb, 0.5, 10.0)

        # Combine scores (simple average of valid metrics)
        valid_metrics = []
        if pe is not None and pe > 0: valid_metrics.append(pe_score)
        if pb is not None and pb > 0: valid_metrics.append(pb_score)
        
        if not valid_metrics:
            return 50.0
        return sum(valid_metrics) / len(valid_metrics)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: higher is better (target 20% as best, 0% as worst)
        roe_score = _linear_score(roe * 100, 0.0, 25.0)
        
        # Debt-to-Equity: lower is better (target 0.2 as best, 1.5 as worst)
        # We invert the logic: higher debt = lower score
        de_score = _linear_score(debt_equity, 1.5, 0.2)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG ratio."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # If PEG is available, it's a strong indicator of growth-at-reasonable-price
        if peg is not None and peg > 0:
            # Map PEG between 0.5 and 3.0 to 100-0
            peg_score = _linear_score(peg, 0.5, 3.0)
            # Combine with ROE for a robust growth/profitability metric
            roe_score = _linear_score(roe * 100, 5.0, 25.0)
            return (peg_score + roe_score) / 2.0
        else:
            # Fallback to ROE-only if PEG is unavailable
            return _linear_score(roe * 100, 5.0, 25.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        
        return sorted_results