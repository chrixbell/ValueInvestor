"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased to reward stable profitability
    "growth": 0.25,    # Increased to capture expansion potential
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

        # Weighted sum aggregation: simpler and less sensitive to extreme low scores.
        composite_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            composite_score += weight * score_val

        if total_weight > 0:
            composite_score /= total_weight
        else:
            composite_score = 50.0

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB → higher score."""
        # Use a mix of PE and PB as they are standard valuation metrics.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback to 20 if PE is None/invalid (neutral-ish)
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Goal: Low PE/PB is good.
        # We use linear mapping to a reasonable range. 
        # PE: 0-30 is good, >50 is bad. PB: 0-5 is good, >10 is bad.
        # Since we want to maximize score:
        pe_score = _linear_score(1.0/pe, 1.0/50.0, 1.0/5.0) if pe > 0 else 0.0
        pb_score = _linear_score(1.0/pb, 1.0/10.0, 1.0/1.0) if pb > 0 else 0.0
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE/Margin (higher better)
        # Scale: 0% to 30% is a common range for healthy companies.
        roe_score = _linear_score(roe, 30.0, -10.0)
        margin_score = _linear_score(margin, 20.0, -5.0)
        
        # Debt (lower better)
        debt_score = max(0.0, min(100.0, (1.0 - (debt / 2.0)) * 100.0)) if debt >= 0 else 50.0

        # Combine
        return (roe_score * 0.4 + margin_score * 0.4 + debt_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG → higher score."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG refinement: low PEG is good for growth-at-reasonable-price.
        if peg is not None and peg > 0:
            peg_score = _linear_score(1.0/peg, 1.0/0.5, 1.0/3.0)
        else:
            peg_score = 50.0

        # ROE is already in quality, but growth-oriented ROE (high) is a signal.
        roe_growth_score = _linear_score(roe, 20.0, -5.0)

        return (peg_score * 0.6 + roe_growth_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results