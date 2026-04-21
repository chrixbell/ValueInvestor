"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased weight to better capture stable businesses
    "growth": 0.25,    # Balanced with quality for a more robust composite
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # We map [0, 100] to [epsilon, 1.0]
            normalized_score = max(0.0001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean: (S1^w1 * S2^w2 * ...) ^ (1 / sum(wi))
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower valuation multiples yield higher scores."""
        # Use PE and PB as primary drivers
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback logic for None values
        if pe is None or pe <= 0:
            pe = 25.0  # Neutral-ish high PE
        if pb is None or pb <= 0:
            pb = 2.0   # Neutral-ish PB

        # Score based on PE (lower is better) and PB (lower is better)
        # We use a simple linear mapping for these widely used metrics
        score_pe = _linear_score(1.0, 40.0, 5.0) if pe > 0 else 0.0
        score_pb = _linear_score(1.0, 4.0, 0.5) if pb > 0 else 0.0
        
        # If PE is extremely high, it might be a growth stock; cap the penalty.
        if pe > 100: score_pe = 0.0

        return (score_pe * 0.6) + (score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage yield higher scores."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE: High is good (target ~20%)
        score_roe = _linear_score(roe * 100, 30.0, -5.0)
        # Debt: Low is good (target ~0.5)
        score_debt = _linear_score(1.0, 0.2, 1.5)
        # Margin: High is good (target ~15%)
        score_margin = _linear_score(margin * 100, 25.0, -5.0)

        return (score_roe * 0.4) + (score_debt * 0.3) + (score_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Growth-oriented scoring."""
        # Using PEG as a primary growth/value hybrid factor
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG: Lower is better (ideal ~1.0)
        if peg <= 0:
            # If PEG is non-positive (likely due to negative earnings), use ROE as proxy
            score_peg = 50.0
        else:
            score_peg = _linear_score(1.0, 1.0, 3.0)

        # ROE is also a measure of growth potential/efficiency
        score_roe = _linear_score(roe * 100, 25.0, 0.0)

        return (score_peg * 0.5) + (score_roe * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results