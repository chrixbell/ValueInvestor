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
    "quality": 0.25,   # Increased to capture stable returns
    "growth": 0.25,    # Increased to capture expansionary potential
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

        # We use an arithmetic mean of normalized scores to prevent a single 0.0 
        # (from a missing field) from zeroing out the entire composite score,
        # while still allowing weights to dictate importance.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # We use the raw score [0, 100] directly in an arithmetic weighted average
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower valuation ratios imply higher score."""
        # Use PE or PB as primary drivers. 
        # If PE is negative (loss), use PB to avoid extreme outliers.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        if pe is not None and pe > 0:
            # Using a standard range for PE (e.g., 1 to 40)
            return _linear_score(pe, 2.0, 35.0) if pe < 100 else 0.0
        elif pb is not None and pb > 0:
            return _linear_score(pb, 0.5, 5.0) if pb < 20 else 0.0
        else:
            # Fallback to dividend yield if PE/PB unavailable
            dy = result.valuation.dividend_yield
            if dy is not None and dy > 0:
                return _linear_score(dy, 2.0, 10.0)
            return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt imply higher score."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE is a primary quality metric
        roe_val = 0.0
        if roe is not None:
            # Map ROE (expecting percentage, e.g., 0.15 for 15%)
            # If ROE is passed as 15.0, we treat it accordingly. 
            # Assuming input scale is consistent with growth_score (e.g., 0.15 or 15.0)
            # We use a simple linear mapping for standard ROE ranges.
            roe_val = _linear_score(roe, 0.05, 0.30) if abs(roe) < 1.0 else _linear_score(roe, 5.0, 30.0)

        # Debt/Equity penalty
        debt_val = 50.0
        if debt_equity is not None:
            # Lower debt is better. 0 is best, 2.0+ is high risk.
            debt_val = _linear_score(debt_equity, 0.0, 2.0)

        return (roe_val * 0.7) + (debt_val * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth (via ROE/PEG) implies higher score."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG is a growth-adjusted valuation metric
        if peg is not None and peg > 0:
            # PEG of 1.0 is fair, < 1.0 is great.
            return _linear_score(peg, 0.5, 2.5)
        elif roe is not None:
            # If no PEG, use ROE as a proxy for growth potential
            roe_val = _linear_score(roe, 0.05, 0.30) if abs(roe) < 1.0 else _linear_score(roe, 5.0, 30.0)
            return roe_val
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results