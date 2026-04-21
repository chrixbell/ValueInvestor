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
    "quality": 0.25,   # Increased to capture more fundamental stability
    "growth": 0.25,    # Increased to capture growth potential
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

        # Using a weighted arithmetic mean instead of geometric to prevent 
        # a single zero-score factor from completely nullifying the result,
        # while still allowing weights to dictate importance.
        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_score += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        v = result.valuation
        f = result.financials
        
        # Use combination of PE and PB for valuation if available
        if v.pe_ratio and v.pe_ratio > 0:
            # We use a target-based approach for linear scoring
            # Low PE is good, but extremely low (near 0) might be distressed.
            # We map PE in range [1, 40] to scores.
            pe_score = _linear_score(v.pe_ratio, 1.0, 40.0)
            # Inverse: lower PE -> higher score
            pe_score = 100.0 - pe_score
        elif v.pb_ratio and v.pb_ratio > 0:
            pb_score = _linear_score(v.pb_ratio, 0.5, 10.0)
            pe_score = 100.0 - pb_score
        else:
            pe_score = 50.0

        # Dividend yield is a strong value signal
        if v.dividend_yield and v.dividend_yield > 0:
            div_score = _linear_score(v.dividend_yield, 0.0, 10.0)
            return (pe_score * 0.7) + (div_score * 0.3)
        
        return pe_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        f = result.financials
        q_score = 0.0
        
        # ROE is primary quality metric
        if f.roe:
            # Map ROE (e.g., -20 to 40) to score
            roe_score = _linear_score(f.roe, -10.0, 30.0)
        else:
            roe_score = 50.0
            
        # Leverage (Debt to Equity)
        if f.debt_to_equity is not None and f.debt_to_equity > 0:
            leverage_score = _linear_score(f.debt_to_equity, 0.0, 2.0)
            leverage_score = 100.0 - leverage_score # Lower debt is better
        else:
            leverage_score = 50.0

        # Margin
        if f.net_margin is not None:
            margin_score = _linear_score(f.net_margin, -5.0, 20.0)
        else:
            margin_score = 50.0

        return (roe_score * 0.5) + (leverage_score * 0.3) + (margin_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        v = result.valuation
        f = result.financials
        
        # Growth often correlates with ROE in these datasets
        if f.roe:
            roe_score = _linear_score(f.roe, 0.0, 30.0)
        else:
            roe_score = 50.0

        # PEG Ratio (Price/Earnings to Growth) - lower is better for growth value
        if v.peg_ratio and v.peg_ratio > 0:
            peg_score = _linear_score(v.peg_ratio, 0.5, 3.0)
            peg_score = 100.0 - peg_score
        else:
            peg_score = 50.0

        return (roe_score * 0.4) + (peg_score * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
        
        return sorted_results