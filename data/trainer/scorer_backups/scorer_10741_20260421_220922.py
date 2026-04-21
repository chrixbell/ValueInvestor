"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture more stable returns
    "growth": 0.25,    # Balanced with quality
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

    # Use a small epsilon to avoid log(0) if input is very close to 0
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

        # Using a weighted arithmetic mean for better stability in ranking 
        # when individual factor scores might be zero.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Prioritize forward_pe if available, then pe_ratio. 
        # Use PB as backstop for asset-heavy industries.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Handle cases where PE might be negative (loss-making)
        if pe is not None and pe > 0:
            # Use a reasonable range for PE (1 to 50)
            s_pe = _linear_score(pe, 1.0, 50.0) if pe < 50 else 0.0
        else:
            s_pe = 0.0

        s_pb = _linear_score(pb, 0.5, 10.0) if pb is not None and pb > 0 else (50.0 if pb is None else 0.0)
        s_ps = _linear_score(ps, 1.0, 20.0) if ps is not None and ps > 0 else (50.0 if ps is None else 0.0)

        # Weighting value components
        return (s_pe * 0.5 + s_pb * 0.3 + s_ps * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin and lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE is a primary quality metric
        s_roe = _linear_score(roe, 0.05, 0.30) if roe is not None else 50.0
        # Margin stability
        s_margin = _linear_score(margin, 0.05, 0.25) if margin is not None else 50.0
        # Leverage (lower is better)
        s_debt = _linear_score(debt, 0.0, 1.5) if debt is not None else 50.0

        return (s_roe * 0.5 + s_margin * 0.3 + s_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        
        # Growth component: PEG (Price/Earnings to Growth) 
        if peg is not None and peg > 0:
            s_peg = _linear_score(peg, 0.5, 2.0)
        else:
            s_peg = 50.0

        # ROE as a proxy for internal growth efficiency
        s_roe = _linear_score(roe, 0.05, 0.30) if roe is not None else 50.0

        return (s_roe * 0.4 + s_peg * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results