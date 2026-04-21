"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced from 0.65 to balance with quality/growth
    "quality": 0.25,   # Increased from 0.15 to prioritize stability
    "growth": 0.25,    # Increased from 0.20 to capture expansion
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
            # Use a small epsilon to prevent division by zero or log issues 
            # if score_val is exactly 0, while still allowing it to drag down the score.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores [0,1] scaled back to [0, 100]
            # For a single factor (weight=1), this is simply score_val.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use pe_forward if available, else pe_ratio
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Fallback if both are None
        if pe is None and pb is None:
            return 50.0

        # Scoring logic: lower PE/PB is better.
        # We use a range-based approach. 
        # For PE: 0-30 is a reasonable range for 'value'.
        # For PB: 0-5 is a reasonable range.
        
        score = 0.0
        if pe is not None and pe > 0:
            # Map PE 0-30 to 100-0. Clamp at 30.
            pe_score = max(0.0, min(100.0, (30.0 - pe) / 30.0 * 100.0))
            score += pe_score
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; usually implies loss. Assign low score but not zero.
            score += 10.0
        else: # pe is None, use pb
            if pb is not None and pb > 0:
                pb_score = max(0.0, min(100.0, (5.0 - pb) / 5.0 * 100.0))
                score += pb_score
            else:
                score += 50.0

        # If both are provided, weight them equally
        if pe is not None and pb is not None:
            return score / 2.0
        return score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: higher is better. Target range -10% to 30%.
        roe_score = _linear_score(roe * 100, 30.0, -10.0)
        
        # Debt/Equity: lower is better. Target range 0 to 1.5.
        de_score = _linear_score(debt_equity, 0.0, 1.5)
        
        # Weight quality factors: ROE is usually a stronger signal.
        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # PEG: lower is better (growth at a reasonable price).
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            peg_score = 50.0 # Neutral for negative PEG (high growth/loss)
        else:
            peg_score = 50.0

        # Margin: higher is better. Target range 0% to 25%.
        margin_score = _linear_score(margin * 100, 25.0, 0.0)

        return (peg_score * 0.6) + (margin_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        # Sort descending by composite_score
        sorted_results = sorted(results, key=lambda r: r.composite_score, reverse=True)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results