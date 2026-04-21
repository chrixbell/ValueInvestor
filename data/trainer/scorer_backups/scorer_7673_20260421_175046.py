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
    "quality": 0.25,   # Increased to reward stable companies
    "growth": 0.25,    # Balanced with quality
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

        # We use a weighted arithmetic mean of normalized scores to prevent 
        # a single zero-score from wiping out the entire composite score, 
        # while still allowing low scores to pull down the average.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += (score_val * weight)

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE is None or negative (though we handle it via clamping)
        pe_val = pe if pe is not None and pe > 0 else 20.0
        pb_val = pb if pb is not None and pb > 0 else 2.0

        # Scoring: lower PE/PB is better.
        # We map a range of PE [0, 30] and PB [0, 10] to [0, 100].
        # Note: This is a simple linear heuristic.
        pe_score = _linear_score(pe_val, 5.0, 30.0)
        pb_score = _linear_score(pb_val, 1.0, 10.0)
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: higher is better (target ~20%)
        roe_score = _linear_score(roe, 15.0, 30.0)
        # Debt to Equity: lower is better (target ~0.5)
        debt_score = _linear_score(debt_eq, 0.1, 1.5)

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG ratio."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        # PEG: lower (closer to 1 or below) is better for growth-at-a-reasonable-price
        # We use 0.5 as best and 3.0 as worst (standard PEG scoring)
        # Since _linear_score expects higher = better, we invert the logic or use a custom mapping.
        # Let's map PEG in [0.1, 3.0] where 0.1 is best (100) and 3.0 is worst (0).
        
        # To use _linear_score(value, best, worst) where best=100:
        # We need to flip the value if we want lower PEG to be higher score.
        # Or simply: score = (worst - value) / (worst - best) * 100
        # Let's use a manual approach for clarity.
        if peg <= 0:
            return 100.0 # Assume very low/negative PEG is great growth potential
        
        # Inverse linear: 0.5 -> 100, 3.0 -> 0
        score = (3.0 - peg) / (3.0 - 0.5) * 100.0
        return max(0.0, min(100.0, score))

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results