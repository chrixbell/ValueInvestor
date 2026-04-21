"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow for more balanced quality/growth
    "quality": 0.25,   # Increased to prioritize stable returns and balance growth
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Use a small epsilon to prevent 0.0 in geometric mean if score_val is 0
            # This allows a single bad factor to penalize but not zero out the whole score
            safe_score = max(0.001, score_val / 100.0)
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # The geometric mean formula for weighted factors is:
            # (S1^w1 * S2^w2 ...)^(1 / sum(weights))
            # But since we want the result in [0, 100], and our base is (S/100),
            # we must multiply the result by 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PEG as a secondary check if available, but keep primary on PE/PB
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback to 20 if PE is negative/None (avoiding outlier noise)
        pe_val = pe if (pe is not None and pe > 0) else 20.0
        pb_val = pb if (pb is not None and pb > 0) else 2.0
        
        # Score based on PE (target range 5-30) and PB (target range 1-5)
        # We use a simple linear interpolation approach for simplicity
        pe_s = _linear_score(pe_val, 5.0, 30.0)
        pb_s = _linear_score(pb_val, 1.0, 5.0)
        
        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        debt = result.financials.debt_to_equity if (result.financials.debt_to_equity is not None) else 1.0
        
        # ROE score (target range -5% to 30%)
        roe_s = _linear_score(roe, 0.25, -0.10)
        # Debt score (target range 0 to 1.5) - lower is better
        debt_s = _linear_score(debt, 0.0, 1.5)
        # Invert debt_s because lower debt should be higher score
        debt_s = 100.0 - debt_s
        
        return (roe_s + debt_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE or growth potential."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        
        # PEG score (target range 0.5 to 2.5) - lower is better
        if peg is not None and peg > 0:
            peg_s = _linear_score(peg, 0.5, 2.5)
            peg_s = 100.0 - peg_s
        else:
            peg_s = 50.0
            
        # ROE can act as a proxy for growth efficiency
        roe_s = _linear_score(roe, 0.30, -0.10)
        
        return (peg_s + roe_s) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for res in results:
            res.score(res)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results