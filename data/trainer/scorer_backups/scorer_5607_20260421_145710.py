"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow room for quality/growth
    "quality": 0.25,   # Increased to prioritize stable earnings and balance sheet strength
    "growth": 0.25,    # Increased to capture expansionary potential
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
            # Using a small epsilon to prevent math errors with zero scores in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Handle the case where product is zero to avoid math errors
            if product_of_powers == 0:
                composite_score = 0.0
            else:
                # The weight normalization is implicitly handled by the powers in product_of_powers
                # If weights sum to 1, (S^w) is already the geometric mean.
                # If weights don't sum to 1, we normalize by applying (1/total_weight)
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB or higher dividend yield is better."""
        # We use a combination of PE and PB for valuation. 
        # If PE is missing, we fall back to PB or Dividend Yield logic.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Base score on PE/PB if available
        if pe is not None and pe > 0:
            # High PE is bad for value
            pe_score = _linear_score(1/pe, 0.01, 40) # Inverse PE (Earnings Yield)
        elif pb is not None and pb > 0:
            pb_score = _linear_score(1/pb, 0.1, 10)
            pe_score = pb_score
        else:
            pe_score = 50.0

        # Dividend yield is a strong value signal
        dy_score = 0.0
        if dy is not None:
            dy_score = _linear_score(dy, 0.02, 0.10)

        return (pe_score * 0.7) + (dy_score * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        q_score = 50.0
        
        # ROE is a primary quality metric
        roe_val = roe if roe is not None else 0.0
        # Scale ROE: typical good range 5% to 30%
        roe_score = _linear_score(roe_val, 0.05, 0.30)
        
        # Leverage: lower is better
        lev_score = 50.0
        if debt_equity is not None:
            # Assume 0 to 2.0 range for debt-to-equity
            lev_score = _linear_score(1.0/max(0.01, debt_equity), 0.2, 2.0)
        
        # Margin: higher is better
        margin_val = margin if margin is not None else 0.0
        margin_score = _linear_score(margin_val, 0.05, 0.25)

        q_score = (roe_score * 0.5) + (lev_score * 0.3) + (margin_score * 0.2)
        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG ratio."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe
        
        # If PEG is available, it's a great growth-at-reasonable-price metric
        if peg is not None and peg > 0:
            # PEG < 1 is good, PEG > 3 is bad
            peg_score = _linear_score(1/peg, 0.5, 3.0)
        else:
            # Fallback to ROE-based growth proxy
            roe_val = roe if roe is not None else 0.0
            peg_score = _linear_score(roe_val, 0.1, 0.4)
            
        return peg_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results