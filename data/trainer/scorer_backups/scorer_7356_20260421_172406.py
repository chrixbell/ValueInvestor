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
    "quality": 0.25,   # Increased to reward stable business models
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
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
            # Use a small epsilon to prevent zero-multiplication in geometric mean
            # while keeping the score relative. 0.01 maps to a very low but non-zero factor.
            adj_score = max(0.01, score_val) / 100.0
            product_of_powers *= (adj_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PEG is better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Use a combination of PE and PB for value, with PEG as a stabilizer
        # We use simple linear logic within the sub-scorer.
        score = 0.0
        count = 0
        
        if pe is not None and pe > 0:
            # PE target range 5-25
            score += _linear_score(pe, 5.0, 30.0)
            count += 1
        if pb is not None and pb > 0:
            score += _linear_score(pb, 1.0, 5.0)
            count += 1
        if peg is not None and peg > 0:
            score += _linear_score(peg, 0.5, 2.0)
            count += 1

        return score / count if count > 0 else 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        net_margin = result.financials.net_margin

        score = 0.0
        count = 0

        if roe is not None:
            score += _linear_score(roe, 5.0, 25.0)
            count += 1
        if net_margin is not None:
            score += _linear_score(net_margin, 5.0, 20.0)
            count += 1
        if debt_to_equity is not None:
            # Lower debt is better. Invert the linear score logic.
            # If debt=0, score=100; if debt=1.5 (150%), score=0
            score += _linear_score(2.0 - debt_to_equity, 0.0, 1.5)
            count += 1

        return score / count if count > 0 else 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and growth-related indicators."""
        # Growth is often captured by PEG in value, but we use ROE and margin here.
        # In many contexts, growth is the delta in revenue/income.
        rev_growth = None # Not explicitly provided as a single field, but we use ROE/Margin
        roe = result.financials.roe
        
        score = 0.0
        count = 0

        if roe is not None:
            # Growth-oriented investors often look for higher ROE
            score += _linear_score(roe, 10.0, 35.0)
            count += 1
        elif result.valuation.peg_ratio is not None:
            # Fallback to PEG if ROE isn't used for growth
            score += _linear_score(result.valuation.peg_ratio, 0.5, 2.0)
            count += 1

        return score / count if count > 0 else 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results