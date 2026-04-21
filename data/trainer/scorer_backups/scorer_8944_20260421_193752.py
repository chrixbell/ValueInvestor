"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced weight on pure value to avoid "value traps"
    "quality": 0.30,   # Increased quality weight to ensure fundamental strength
    "growth": 0.30,    # Balanced growth with quality
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
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
        self.weights = weights or dict(_DEFAULT_weights := _DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: Screening_Result) -> Screening_Result:
        """Compute sub-scores and composite score, updating *result* in place."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        # Momentum is a placeholder
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

        # Using an additive weighted average for stability across different factor distributions
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

    def _value_score(self, result: Screening_Result) -> float:
        """Lower PE/PB/PS is better."""
        # Prioritize Forward PE over Trailing PE if available
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Handle potential negative PEs/PBs
        pe_score = 0.0
        if pe is not None and pe > 0:
            pe_score = _linear_score(pe, 10, 40) # Target PE range
        
        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1, 5)

        ps_score = 0.0
        if ps is not None and ps > 0:
            ps_score = _linear_score(ps, 1, 5)

        return (pe_score * 0.4 + pb_score * 0.3 + ps_score * 0.3)

    def _quality_score(self, result: Screening_Result) -> float:
        """Higher ROE/Margin and lower leverage is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE/Margin: Higher is better (using linear scale)
        roe_score = max(0.0, min(100.0, (roe + 0.2) * 100)) # Assuming ROE is fraction
        margin_score = max(0.0, min(100.0, (margin + 0.2) * 100))
        
        # Debt: Lower is better
        debt_score = _linear_score(debt, 0.1, 1.5)
        
        return (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.3)

    def _growth_score(self, result: Screening_Result) -> float:
        """Higher ROE and lower PEG is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        
        # Use ROE as a proxy for growth capability if PEG is missing
        if peg is not None and peg > 0:
            # Low PEG is good (e.g., 0.5-1.5)
            peg_score = _linear_score(peg, 0.5, 3.0)
            # ROE is still a quality/growth component
            roe_score = max(0.0, min(100.0, (roe + 0.2) * 100))
            return peg_score * 0.7 + roe_score * 0.3
        else:
            return max(0.0, min(100.0, (roe + 0.2) * 100))

    def rank(self, results: List[Screening_Result]) -> List[Screening_Result]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results