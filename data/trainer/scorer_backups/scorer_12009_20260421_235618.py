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
    "quality": 0.25,   # Increased quality weight to ensure robust earnings
    "growth": 0.25,    # Balanced with quality for a sustainable factor mix
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

        product_of_powers = 1.0
        total_weight = 0.0

        # To prevent a single zero score from wiping out the entire composite via geometric mean,
        # we use a small epsilon to ensure the score is at least 0.01 if other factors are strong,
        # but we still allow the zero to penalize. However, for pure geometric mean, 
        # we must handle the case where score_val is 0.
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent log(0) or zeroing out the entire product if one factor is 0
            # but still allow it to be a significant penalty.
            normalized_val = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product_of_powers is (S1/100)^w1 * (S2/100)^w2...
            # The geometric mean is product^(1/sum_weights). 
            # Since we already applied weights in the exponent, we don't divide by total_weight again 
            # if we want the result to stay in [0, 100] range naturally.
            # Actually, the formula for weighted geometric mean is: exp( sum(w_i * ln(x_i)) / sum(w_i) )
            # Our product_of_powers is already the numerator of that.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluates value based on PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None or invalid PE (e.g., negative)
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Map PE (lower is better) to score [0, 100]
            # Using a wide range for PE: 0 to 40.
            pe_score = _linear_score(pe, best=1.0, worst=40.0)

        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Map PB (lower is better) to score [0, 100]
            pb_score = _linear_score(pb, best=0.5, worst=10.0)

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluates quality based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE (higher is better)
        roe_score = _linear_score(roe, best=0.25, worst=-0.1)
        
        # Debt/Equity (lower is better)
        de_score = _linear_score(debt_equity, best=0.1, worst=2.0)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluates growth based on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None

        # ROE component (higher is better)
        roe_score = _linear_score(roe, best=0.20, worst=-0.05)

        if peg is None or peg <= 0:
            # If PEG is unavailable, rely on ROE
            return roe_score
        else:
            # PEG (lower is better)
            peg_score = _linear_score(peg, best=0.5, worst=3.0)
            return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sorts results by composite score and assigns ranks."""
        # Sort descending (higher score = better rank)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results