"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight to quality/growth
    "quality": 0.25,   # Increased quality weighting to filter out "value traps"
    "growth": 0.25,    # Balanced growth weight
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

        # To prevent a single zero score from zeroing out the entire composite via 
        # geometric mean (which can happen with bad data), we use a small epsilon.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Add epsilon to prevent product becoming zero if a sub-score is 0
            normalized_score = (max(epsilon, score_val) / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The formula (prod(s^w))^(1/sum_w) effectively calculates the weighted geometric mean
            # of the normalized scores. We scale it back to 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PEG as a tie-breaker/modifier if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Fallback logic for None values
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        peg = peg if peg is not None and peg > 0 else 1.5

        # Primary score: combination of PE and PB
        # We use a simple linear mapping for simplicity in this iteration
        score_pe = _linear_score(pe, 10.0, 40.0)
        score_pb = _linear_score(pb, 1.0, 5.0)
        
        # PEG is a growth-adjusted value metric (lower is better)
        score_peg = _linear_score(peg, 0.5, 3.0)

        return (score_pe * 0.4 + score_pb * 0.4 + score_peg * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE score (target ~15-20%)
        score_roe = _linear_score(roe * 100, 5.0, 30.0)
        # Leverage score (lower is better)
        score_leverage = _linear_score(debt_to_equity, 0.1, 1.5)
        # Margin score
        score_margin = _linear_score(margin * 100, 5.0, 25.0)

        return (score_roe * 0.4 + score_leverage * 0.3 + score_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.5
        
        # Growth focus: ROE as a proxy for internal growth capability
        score_roe = _linear_score(roe * 100, 5.0, 30.0)
        # PEG as a proxy for growth attractiveness
        score_peg = _linear_score(peg, 0.5, 2.5)

        return (score_roe * 0.6 + score_peg * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results