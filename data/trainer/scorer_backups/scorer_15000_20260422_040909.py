"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to balance with quality/growth
    "quality": 0.30,   # Increased: high-quality companies often have more stable returns
    "growth": 0.20,    # Maintained growth component
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

        # We use an arithmetic mean of normalized scores to avoid the extreme 
        # sensitivity of geometric means when a single sub-score is zero.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # To prevent a single zero-score from wiping out the entire composite (as in geometric mean),
            # we use a weighted arithmetic average of the [0, 100] scores.
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        v = result.valuation
        f = result.financials
        
        # Primary: Forward PE (if available) or Trailing PE
        pe = v.pe_forward if v.pe_forward is not None and v.pe_forward > 0 else v.pe_ratio
        pb = v.pb_ratio
        ps = v.ps_ratio
        div = v.dividend_yield

        # Fallback values for ranking logic
        pe = pe if (pe is not None and pe > 0) else 20.0
        pb = pb if (pb is not None and pb > 0) else 2.0
        ps = ps if (ps is not None and ps > 0) else 2.0
        div = div if (div is not None and div > 0) else 0.0

        # Scoring components
        s_pe = _linear_score(pe, 30.0, 5.0)
        s_pb = _linear_score(pb, 3.0, 0.5)
        s_ps = _linear_score(ps, 3.0, 0.5)
        s_div = _linear_score(div if div > 0 else 1.0, 5.0, 0.0)

        # Combine: PE is the heaviest weight for value
        return (s_pe * 0.5) + (s_pb * 0.2) + (s_ps * 0.1) + (s_div * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/Margin, lower leverage."""
        f = result.financials
        roe = f.roe if f.roe is not None else 0.0
        margin = f.net_margin if f.net_margin is not None else 0.0
        debt = f.debt_to_equity if f.debt_to_equity is not None else 0.5
        roa = f.roa if f.roa is not None else 0.0

        # ROE and Margin are key quality indicators
        s_roe = _linear_score(roe, 0.25, -0.1)
        s_margin = _linear_score(margin, 0.20, -0.1)
        s_roa = _linear_score(roa, 0.15, -0.05)
        # Debt: lower is better
        s_debt = _linear_score(debt, 0.5, 0.0)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_roa * 0.2) + (s_debt * 0.1)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        v = result.valuation
        f = result.financials
        roe = f.roe if f.roe is not None else 0.0
        peg = v.peg_ratio if (v.peg_ratio is not None and v.peg_ratio > 0) else 2.0

        # Growth-Value interaction: PEG is a measure of growth at a reasonable price
        s_roe = _linear_score(roe, 0.25, -0.1)
        s_peg = _linear_score(peg, 1.0, 0.5) # Lower PEG is better

        return (s_roe * 0.4) + (s_peg * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort results by composite score and assign ranks."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results