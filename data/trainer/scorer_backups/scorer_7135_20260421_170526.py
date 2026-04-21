"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to provide more room for quality/growth
    "quality": 0.30,   # Increased to capture more stable earnings-per-share drivers
    "growth": 0.20,    # Retained to capture upside potential
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
            # Using a small epsilon to prevent math.log(0) or issues with 0 scores in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # In a geometric mean, if any score is 0, the product is 0.
            # We use a tiny epsilon to allow very low scores to not zero out everything, 
            # but since we are using the product directly:
            if product_of_powers == 0 and total_weight > 0:
                # If we hit a zero, the geometric mean is 0. 
                # However, to avoid losing all information if one factor is zero:
                # We could use an arithmetic mean, but we'll stick to the requested structure.
                composite_score = 0.0
            else:
                # The formula (product^(1/sum_w)) * 100 is slightly different for weighted geometric mean.
                # The standard weighted geometric mean is product(x_i^w_i)^(1/sum(w_i)).
                # Since our product is already (S1/100)^w1 * (S2/100)^w2..., 
                # the result is already correctly scaled to [0, 1].
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Default values to avoid None issues in comparison
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Score based on low PE/PB (lower is better)
        # We use a simple linear mapping for the range. 
        # Let's assume "best" PE is 5 and "worst" is 30.
        # Let's assume "best" PB is 1 and "worst" is 6.
        pe_s = _linear_score(30.0 - pe, 30.0 - 5.0, 30.0 - 30.0) # wait, standardizing logic:
        # Let's use simpler logic: 
        # If pe=5, score=100. If pe=30, score=0.
        pe_s = _linear_score(pe, 30.0, 5.0) # This is wrong direction in _linear_score
        # Correcting: _linear_score(value, best, worst) -> if value=best, returns 100.
        # If pe is small (5), it should be 100.
        pe_s = _linear_score(pe, 5.0, 30.0) # No, this would make low PE = 0.
        # Re-implementing logic:
        if pe <= 5.0: pe_s = 100.0
        elif pe >= 30.0: pe_s = 0.0
        else: pe_s = (30.0 - pe) / (30.0 - 5.0) * 100.0

        if pb <= 0.5: pb_s = 100.0
        elif pb >= 5.0: pb_s = 0.0
        else: pb_s = (5.0 - pb) / (5.0 - 0.5) * 100.0

        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE: Higher is better (e.g., 25% is best, 0% is worst)
        roe_s = _linear_score(roe, 25.0, -5.0)
        # Debt: Lower is better (e.g., 0.2 is best, 1.5 is worst)
        debt_s = _linear_score(debt, 0.2, 1.5)
        
        return (roe_s + debt_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Computes growth score using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 2.0

        # High ROE is good
        roe_s = _linear_score(roe, 20.0, -10.0)
        # Low PEG is good (Growth at reasonable price)
        if peg <= 0.5: peg_s = 100.0
        elif peg >= 3.0: peg_s = 0.0
        else: peg_s = (3.0 - peg) / (3.0 - 0.5) * 100.0

        return (roe_s + peg_s) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results