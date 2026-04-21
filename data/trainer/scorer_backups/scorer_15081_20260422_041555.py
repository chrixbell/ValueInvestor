"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to prioritize more stable companies
    "growth": 0.25,    # Balanced with quality to capture profitable growth
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

        # Use an arithmetic mean of normalized scores to prevent a single zero 
        # sub-score from wiping out the entire composite score, which is more 
        # robust for real-world data where one factor might be missing.
        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # We use the linear scale [0, 100] directly for the weighted average
            total_weighted_score += score_val * weight

        if total_weight > 0:
            # Calculate the weighted arithmetic mean
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a mix of valuation metrics. 
        # We prioritize PE if available, otherwise PB.
        v_pe = result.valuation.pe_ratio
        v_pb = result.valuation.pb_ratio
        v_ps = result.valuation.ps_ratio

        # Target ranges for 'best' and 'worst' to avoid extreme outliers
        # PE: 5 (best) to 30 (worst)
        # PB: 1 (best) to 10 (worst)
        # PS: 1 (best) to 15 (worst)

        if v_pe is not None and v_pe > 0:
            return _linear_score(v_pe, 5.0, 30.0)
        elif v_pb is not None and v_pb > 0:
            return _linear_score(v_pb, 1.0, 10.0)
        elif v_ps is not None and v_ps > 0:
            return _linear_score(v_ps, 1.0, 15.0)
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        score = 50.0
        
        # ROE component (Target: 20% best, 0% worst)
        if roe is not None:
            # Convert percentage to decimal if needed, but assuming 0.20 format
            roe_val = roe * 100 if roe < 5.0 else roe # Handle both 0.2 and 20.0
            # Using a simple linear mapping for ROE
            roe_s = _linear_score(roe_val, 20.0, -5.0)
            score = score * 0.6 + roe_s * 0.4

        # Debt component (Target: 0.2 best, 2.0 worst)
        if debt_equity is not None:
            debt_s = _linear_score(debt_equity, 0.2, 2.0)
            score = score * 0.4 + debt_s * 0.6
        else:
            # If no debt data, weight ROE more heavily
            score = score * 0.4 + (50.0 * 0.6)

        return score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        score = 50.0
        if peg is not None and peg > 0:
            # PEG: 0.5 (best) to 3.0 (worst)
            peg_s = _linear_score(peg, 0.5, 3.0)
            return peg_s
        elif roe is not None:
            roe_val = roe * 100 if roe < 5.0 else roe
            return _linear_score(roe_val, 15.0, -5.0)
        
        return score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort results by composite score and assign ranks."""
        # First, ensure all are scored
        for r in results:
            if not hasattr(r, 'composite_score'):
                self.score(r)

        # Sort descending by composite score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)

        # Assign ranks (1 is best)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        
        return sorted_results