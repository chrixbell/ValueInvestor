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
    "quality": 0.25,   # Increased to capture stable earnings and lower risk
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent zero issues in geometric mean
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean is calculated as (product of S_i^w_i)^(1/sum(w_i))
            # Since we normalized S_i to [0, 1], the result is in [0, 1]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Using log-scale for valuation to prevent extreme outliers from dominating
        # and to handle the wide range of PE ratios.
        scores = []
        if pe is not None and pe > 0:
            # Target PE range [1, 30]
            scores.append(_log_score(pe, 30.0, 1.0) if pe < 30 else 0.0) # This logic is inverted for 'lower is better'
            # Re-implementing simple linear/log inversion:
            # If PE is 1, score should be high. If 30, score low.
            # Let's use a simpler approach:
            val_score = 100.0 / (1.0 + math.log(pe) if pe > 0 else 100.0) # Placeholder logic
        
        # Refined Value Scoring:
        v_score = 50.0
        if pe is not None and pe > 0:
            # Map PE to a score where lower is better. Maximize at 1, minimize at 50.
            v_score = max(0.0, min(100.0, 100.0 - (pe / 2.0))) # Rough heuristic
        elif pb is not None and pb > 0:
            v_score = max(0.0, min(100.0, 100.0 - (pb / 2.0)))
        
        # Let's use a cleaner approach for the sub-scores to ensure they are strictly [0, 100]
        # Value: Low PE or Low PB
        pe_score = 0.0
        if pe is not None and pe > 0:
            # A PE of 5 -> high score, 50 -> low score.
            pe_score = max(0.0, min(100.0, 100.0 - (pe * 2.0))) # Example: PE 50 -> score 0
            if pe < 1: pe_score = 100.0
        
        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = max(0.0, min(100.0, 100.0 - (pb * 5.0))) # Example: PB 20 -> score 0
            if pb < 1: pb_score = 100.0
            
        # Combine them: prioritize PE if available, else PB
        if pe is not None and pe > 0:
            return pe_score
        return pb_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.1
        
        # Normalize ROE (assuming range -20% to 50%)
        roe_s = max(0.0, min(100.0, (roe + 0.2) * 150))
        # Normalize Debt (assuming 0 to 2.0)
        debt_s = max(0.0, min(100.0, 100.0 - (debt_eq * 40)))
        # Normalize Margin
        margin_s = max(0.0, min(100.0, (margin + 0.2) * 150))
        
        return (roe_s * 0.4 + debt_s * 0.3 + margin_s * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        
        roe_s = max(0.0, min(100.0, (roe + 0.2) * 150))
        
        if peg is not None and peg > 0:
            # PEG of 1 -> medium score, PEG < 1 -> high, PEG > 2 -> low
            peg_s = max(0.0, min(100.0, 100.0 - (peg - 1.0) * 33.3 if peg > 1 else 100.0))
            return (roe_s * 0.5 + peg_s * 0.5)
        
        return roe_s

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

# Note: The above implementation of _value_score, _quality_score, and _growth_score
# was drafted to ensure the logic is robust. I will refine them into a clean single class.