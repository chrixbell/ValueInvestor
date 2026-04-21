"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Adjusted to balance against quality/growth
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
            # We use a small epsilon to prevent log(0) issues if any score is 0
            # but since we use power, (score/100)^weight is fine as long as score >= 0.
            # We use a tiny offset to ensure zero scores don't zero out the whole product if weight is high.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The weighted geometric mean formula is (Product of S_i^w_i)^(1 / Sum w_i)
            # However, since the weights in _DEFAULT_WEIGHTS sum to 1.0 (mostly),
            # we can simplify, but the formula below is robust to any total_weight.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None or negative PE/PB (though PB is usually positive)
        if pe is None or pe <= 0:
            pe_score = 50.0 # Neutral if no valid PE
        else:
            # We want low PE/PB to be high score. 
            # Typical healthy range: PE 5-20, PB 1-3.
            pe_score = _linear_score(pe, best=8.0, worst=30.0)
            
        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            pb_score = _linear_score(pb, best=1.5, worst=5.0)

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # Higher ROE is better
        roe_score = _linear_score(roe, best=20.0, worst=5.0)
        # Lower Debt/Equity is better
        de_score = _linear_score(debt_equity, best=0.2, worst=1.5)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        gm = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0

        # PEG: Lower is better (growth relative to valuation)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, best=0.5, worst=2.5)
        elif peg is not None and peg <= 0:
            peg_score = 100.0 # Negative PEG is very strong growth relative to PE
        else:
            peg_score = 50.0

        # Gross Margin as a proxy for growth/moat stability
        gm_score = _linear_score(gm, best=40.0, worst=10.0)

        return (peg_score + gm_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results