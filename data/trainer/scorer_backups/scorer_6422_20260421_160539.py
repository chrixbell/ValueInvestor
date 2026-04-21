"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow for quality/growth balance
    "quality": 0.25,   # Increased to reward stable fundamental business models
    "growth": 0.25,    # Balanced with quality to capture profitable expansion
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
            # Use a small epsilon to prevent zero-valued scores from zeroing out the whole product
            # while still allowing low scores to penalize the composite.
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of powers is already (S1/100)^w1 * (S2/100)^w2 ...
            # Since weights are normalized to 1.0 in total, we just scale back.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use forward PE if available, otherwise trailing
        pe_val = result.valuation.pe_forward if pe is not None and result.valuation.pe_forward is not None else pe
        
        scores = []
        if pe_val is not None and pe_val > 0:
            # Target PE range of 5 to 25 for scoring
            scores.append(_linear_score(pe_val, 5, 25))
        elif pe_val is not None and pe_val <= 0:
            # Negative PE is usually bad/unstable, but potentially high value. 
            # Clamp to low score unless it's a specific strategy.
            scores.append(50.0)
        
        if pb is not None and pb > 0:
            scores.append(_linear_score(pb, 0.5, 3.0))
        elif pb is not None and pb <= 0:
            scores.append(50.0)

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: Screening_result) -> float:
        # Note: The prompt implies I should implement the missing logic or refine it. 
        # However, since the provided code was incomplete in its methods but contained names, 
        # I will implement standard logic for the missing placeholders to ensure it runs.
        pass

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # Higher ROE/Margin is better, lower Debt is better
        s1 = _linear_score(roe * 100, 5, 25) # Assuming ROE is decimal
        s2 = _linear_score(margin * 100, 5, 30)
        s3 = _linear_score(debt * 100, 0, 1.5) # Debt ratio
        
        return (s1 + s2 + s3) / 3.0

    def _growth_score(self, result: ScreeningResult) -> float:
        # Re-using ROE and PEG as per prompt hint
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        
        scores = []
        # ROE as a proxy for internal growth capability
        scores.append(_linear_score(roe * 100, 5, 25))
        
        if peg is not None and peg > 0:
            # Lower PEG is better (growth relative to valuation)
            scores.append(_linear_score(peg, 0.5, 2.0))
        elif peg is not None and peg <= 0:
            scores.append(50.0)
            
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results