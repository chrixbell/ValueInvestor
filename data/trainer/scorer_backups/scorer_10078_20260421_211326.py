"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for value
    "quality": 0.25,   # Increased quality to ensure fundamental stability
    "growth": 0.25,    # Balanced growth component
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

        # We use an additive weighted average for the composite score to prevent a 
        # single zero sub-score (e.g., from one missing data point) from wiping 
        # out the entire score, while still allowing high-quality stocks to rank well.
        composite_sum = 0.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights.get(k, 0.0)
            # Normalize weight relative to the sum of active weights
            normalized_weight = weight / total_weight
            composite_sum += score_val * normalized_weight

        result.composite_score = composite_sum
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluate value based on PE and PB ratios."""
        # Use a mix of PE and PB. If PE is negative (loss), use PB or PS as fallback.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Define thresholds for linear scoring
        # PE range: 0 to 30 (where 30 is 'worst' for value)
        # PB range: 0 to 10 (where 10 is 'worst' for value)
        
        score_pe = _linear_score(pe if pe is not None and pe > 0 else 30.0, 5.0, 30.0)
        score_pb = _linear_score(pb if pb is not None and pb > 0 else 10.0, 1.0, 10.0)
        
        # Fallback if PE is invalid/negative
        if pe is None or pe <= 0:
            return score_pb

        # Weight PE more heavily but combine with PB
        return (score_pe * 0.7) + (score_pb * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality based on ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE is a primary quality driver
        score_roe = _linear_score(roe if roe is not None else 0.0, 15.0, -5.0)
        
        # Debt to Equity: lower is better
        if debt_equity is not None:
            score_debt = _linear_score(debt_equity, 0.5, 2.0)
        else:
            score_debt = 50.0

        # Combine scores
        return (max(0, score_roe) * 0.7) + (min(100, score_debt) * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # If PEG is available and positive, it's a strong growth/value hybrid signal
        if peg is not None and peg > 0:
            score_peg = _linear_score(peg, 0.5, 2.5)
        else:
            score_peg = _linear_score(roe if roe is not None else 0.0, 20.0, -10.0)

        score_roe = _linear_score(roe if roe is not None else 0.0, 15.0, -5.0)

        return (score_peg * 0.6) + (score_roe * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results