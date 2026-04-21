"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slight reduction to balance with quality/growth
    "quality": 0.25,   # Increased weight for quality to provide a stable baseline
    "growth": 0.25,    # Balanced with quality
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

        # We use a weighted arithmetic mean of normalized scores to prevent a single 
        # zero-score from wiping out the entire composite (which occurs in geometric mean).
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] for aggregation
            total_weighted_sum += (score_val / 100.0) * weight

        if total_weight > 0:
            # Calculate the weighted arithmetic mean and scale back to [0, 100]
            composite_score = (total_weighted_sum / total_weight) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score. Lower PE/PB is better."""
        # Use PE and PB as primary metrics. If one is missing, fallback to others.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        
        # Define common ranges for A-share/HK stocks
        # We use a simple heuristic: higher score for lower ratios.
        # To handle the scale, we map typical ranges to 0-100.
        
        # PE score: range [0, 50] -> low values are good.
        pe_score = 100.0
        if pe is not None and pe > 0:
            # A PE of 1 is very low (score ~98), a PE of 40 is high (score ~20)
            pe_score = max(0.0, min(100.0, 100.0 - (pe / 40.0 * 80.0)))
        elif pe is not None and pe <= 0:
            pe_score = 100.0 # Negative PE is often better for value (though tricky)
            
        # PB score: range [0, 30]
        pb_score = 100.0
        if pb is not None and pb > 0:
            pb_score = max(0.0, min(100.0, 100.0 - (pb / 5.0 * 80.0)))
        elif pb is not None and pb <= 0:
            pb_score = 100.0

        # PS score fallback
        ps_score = 100.0
        if ps is not None and ps > 0:
            ps_score = max(0.0, min(100.0, 100.0 - (ps / 5.0 * 80.0)))

        # Combine scores
        if pe is not None and pb is not None:
            return (pe_score + pb_score) / 2.0
        elif pe is not None:
            return pe_score
        elif pb is not None:
            return pb_score
        else:
            return ps_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score. Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE component (Targeting 0% to 30%)
        roe_score = 50.0
        if roe is not None:
            # Map ROE 25% -> 100, ROE -5% -> 0
            roe_score = max(0.0, min(100.0, (roe + 5.0) / 30.0 * 100.0))

        # Debt component (Targeting 0% to 100%)
        debt_score = 50.0
        if debt_equity is not None:
            # Lower debt is better. 100% debt/equity -> 20 score, 0% -> 100
            debt_score = max(0.0, min(100.0, 100.0 - (debt_equity / 1.5 * 80.0)))

        # Margin component
        margin_score = 50.0
        if margin is not None:
            # Net margin 20% -> 100, -5% -> 0
            margin_score = max(0.0, min(100.0, (margin + 5.0) / 25.0 * 100.0))

        # Weighting quality components
        if roe is not None and debt_equity is not None:
            return (roe_score * 0.6 + debt_score * 0.4)
        elif roe is not None:
            return roe_score
        else:
            return debt_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score. High ROE and low PEG is better."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # ROE is already in quality, but can be used for growth potential if high
        roe_score = 50.0
        if roe is not None:
            roe_score = max(0.0, min(100.0, (roe + 5.0) / 35.0 * 100.0))

        # PEG score: Low PEG is better (growth at reasonable price)
        peg_score = 50.0
        if peg is not None and peg > 0:
            # PEG 0.5 -> 100, PEG 3.0 -> 0
            peg_score = max(0.0, min(100.0, (3.5 - peg) / 3.0 * 100.0))
        elif peg is not None and peg <= 0:
            peg_score = 100.0

        if roe is not None and peg is not None:
            return (roe_score * 0.4 + peg_score * 0.6)
        elif peg is not None:
            return peg_score
        else:
            return roe_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks (1 to N)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results