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
    "quality": 0.25,   # Increased to prioritize stable earners
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

        total_weight = sum(self.weights[k] for k in weighted_scores_to_process)
        
        # We use an arithmetic mean of normalized scores to ensure we don't 
        # zero out the entire composite score if a single factor is 0 (unless that's intended).
        # However, to keep the "quality" check intact, we use a weighted sum.
        total_score = 0.0
        for k, score_val in weighted_scores_to_process.items():
            total_score += score_val * self.weights[k]

        # Normalize by total weight to keep it in [0, 100]
        if total_weight > 0:
            composite_score = total_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PEG and PB as they are often more stable in different market cycles
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Fallback mechanism to ensure we always have a base score
        if pe is None or pe <= 0:
            pe = 20.0 # Neutral default
        if pb is None or pb <= 0:
            pb = 2.0  # Neutral default
        if peg is None or peg <= 0:
            peg = 1.0  # Neutral default

        # Score components (normalized)
        # PE: 5 to 30 is a reasonable range for value
        s_pe = _linear_score(pe, 5.0, 30.0)
        # PB: 0.5 to 4.0 is a reasonable range
        s_pb = _linear_score(pb, 0.5, 4.0)
        # PEG: lower is better (1.0 neutral, <1 good, >2 bad)
        s_peg = _linear_score(peg, 0.5, 2.5)

        return (s_pe * 0.4 + s_pb * 0.3 + s_peg * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and margin, lower leverage."""
        roe = result.financials.roe or 0.0
        margin = result.financials.net_margin or 0.0
        debt_equity = result.financials.debt_to_equity or 1.0 # Default to neutral leverage

        # ROE: 5% to 25%
        s_roe = _linear_score(roe, 0.05, 0.25)
        # Margin: 0% to 20%
        s_margin = _linear_score(margin, 0.0, 0.20)
        # Debt/Equity: Lower is better (0 to 1.5 range)
        s_debt = _linear_score(debt_equity, 1.5, 0.0)

        return (s_roe * 0.4 + s_margin * 0.3 + s_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        # Using a mix of profitability as proxy for growth potential
        roe = result.financials.roe or 0.0
        peg = result.valuation.peg_ratio or 1.0

        # ROE is a strong indicator of growth efficiency
        s_roe = _linear_score(roe, -0.05, 0.30)
        # PEG: Low is better for growth investors
        s_peg = _linear_score(peg, 0.5, 2.0)

        return (s_roe * 0.5 + s_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for idx, r in enumerate(results):
            r.rank = idx + 1
        return results