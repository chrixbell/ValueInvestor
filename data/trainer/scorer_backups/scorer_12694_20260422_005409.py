"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to balance with quality/growth
    "quality": 0.25,   # Increased to emphasize fundamental stability
    "growth": 0.25,    # Balanced with quality to capture high-quality growth
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

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0
        
        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


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

        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        # Using a weighted arithmetic mean for the final aggregation to prevent 
        # a single zero score from wiping out the entire composite (which can happen in geometric mean).
        # However, we still use the weighted sum logic to maintain the relative importance.
        total_weighted_sum = 0.0
        for k, score_val in weighted_scores_to_process.items():
            total_weighted_sum += score_val * self.weights[k]

        composite_score = total_weighted_sum / total_weight
        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score based on PE and PB ratios."""
        # Use PE and PB as primary metrics. 
        pe = result.valuation.pe_ratio if result.valuation.pe_ratio is not None else None
        pb = result.valuation.pb_ratio if result.valuation.pb_ratio is not None else None
        
        # Fallback to forward PE if trailing is missing
        if pe is None:
            pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else None
        
        # Handle cases where PE might be negative (loss-making)
        pe_val = pe if (pe is not None and pe > 0) else 100.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0

        # Scoring: Lower PE/PB is better.
        # We use a wide range for the linear scoring to capture value opportunities.
        score_pe = _linear_score(pe_val, best=5.0, worst=40.0)
        score_pb = _linear_score(pb_val, best=1.0, worst=10.0)
        
        return (score_pe * 0.6) + (score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE: Higher is better. (Assume range -20% to 40%)
        # We map ROE from [-0.2, 0.4] to [0, 100]
        roe_mapped = ((roe + 0.2) / 0.6) * 100.0
        score_roe = max(0.0, min(100.0, roe_mapped))
        
        # Debt/Equity: Lower is better. (Assume range 0 to 2.0)
        score_leverage = _linear_score(debt_equity, best=0.2, worst=1.5)
        
        return (score_roe * 0.7) + (score_leverage * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # If PEG is available, it's a strong growth/value hybrid indicator.
        if peg is not None and peg > 0:
            # PEG < 1 is excellent.
            score_peg = _linear_score(peg, best=0.5, worst=3.0)
        else:
            score_peg = 50.0 # Neutral if PEG is unavailable

        # ROE as a proxy for internal growth/efficiency
        roe_mapped = ((roe + 0.2) / 0.6) * 100.0
        score_roe = max(0.0, min(100.0, roe_mapped))
        
        return (score_peg * 0.5) + (score_roe * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks (1 to N)
        for i, res in enumerate(results):
            res.rank = i + 1
            
        return results