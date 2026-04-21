"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture fundamental stability
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

        total_weight = sum(self.weights[k] for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        # We use a weighted arithmetic mean of the sub-scores for more stable ranking
        # when dealing with zero-valued sub-scores (which can happen in fundamental data).
        total_weighted_sum = 0.0
        for k, score_val in weighted_scores_to_process.items():
            total_weighted_sum += score_val * self.weights[k]

        # Normalize the sum by total weight to get a score in [0, 100]
        # Note: if total_weight is the sum of all weights in _DEFAULT_WEIGHTS, 
        # we normalize by that to ensure the score stays within [0, 100].
        # However, since we only iterate over active weights:
        composite_score = total_weighted_sum / (total_weight / sum(self.weights.values()) if sum(self.weights.values()) != 0 else 1)
        # To ensure it stays in [0, 100] regardless of weight distribution:
        # The math above simplifies to a standard weighted average if we assume 
        # the weights provided are absolute. Let's use a cleaner weighted average:
        
        final_score = 0.0
        current_total_weight = 0.0
        for k, score_val in weighted_scores_to_process.items():
            w = self.weights[k]
            final_score += score_val * w
            current_total_weight += w
        
        if current_total_weight > 0:
            result.composite_score = final_score / current_total_weight
        else:
            result.composite_score = 50.0

        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is generally better for value."""
        # Use a mix of PE and PB if available, else fallback to single metric
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        eps = result.financials.net_income / result.valuation.price if (result.valuation.price and result.financials.net_income) else None
        
        # If we have a valid PE or PB, use them. 
        # We use high-end caps to prevent extreme outliers from skew0ng the score.
        if pe and pe > 0:
            return _linear_score(pe, 5.0, 40.0) if pe < 40 else 0.0
        if pb and pb > 0:
            return _linear_score(pb, 1.0, 10.0) if pb < 10 else 0.0
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # Score ROE (higher better)
        roe_s = _linear_score(roe, 5.0, 25.0) if roe != 0 else 50.0
        # Score Debt/Equity (lower better) - Clamp to avoid division errors or negative logic
        # We use a simple linear mapping for debt.
        debt_s = _linear_score(1/max(0.01, debt_equity), 0.1, 2.0) if debt_equity != 0 else 50.0
        if debt_equity == 0: debt_s = 100.0

        return (roe_s * 0.7) + (debt_s * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio
        
        # PEG is a strong growth/value hybrid metric
        if peg and peg > 0:
            # Low PEG is high score. Clamp at 4.0 to avoid extreme scores.
            peg_s = _linear_score(1/peg, 0.5, 4.0) # This is not quite right for linear_score
            # Let's use direct mapping:
            peg_s = max(0.0, min(100.0, (4.0 - peg) / 4.0 * 100.0)) if peg < 4.0 else 0.0
            return peg_s
        
        # Fallback to ROE if PEG is unavailable
        return _linear_score(roe, 5.0, 30.0) if roe != 0 else 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results