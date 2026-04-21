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
    "quality": 0.25,   # Increased to emphasize stable fundamentals
    "growth": 0.25,    # Increased to capture expansion potential
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
            # We use a small epsilon to prevent 0 score from zeroing out the whole product via geometric mean
            # This ensures that a single bad factor doesn't destroy the score unless it is truly catastrophic
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers already incorporates the weights. 
            # (S1^w1 * S2^w2) is the product. To scale back to [0, 100], we don't need a second root
            # unless the weights were not normalized to sum to 1. 
            # However, for consistency with the previous structure:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use PEG as a tie-breaker/supplement if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Handle None values by providing neutral defaults
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        peg = peg if peg is not None and peg > 0 else 1.5

        # Value score: Lower PE/PB is better
        # We use a simple linear mapping for basic valuation components
        score_pe = _linear_score(20.0 / pe, 20.0, 2.0) # Lower PE -> higher score
        score_pb = _linear_score(2.0 / pb, 2.0, 0.5)  # Lower PB -> higher score
        
        # If PEG is available, it's a powerful value/growth hybrid
        if peg:
            score_peg = _linear_score(1.0 / peg, 2.0, 0.5)
            return (score_pe * 0.4 + score_pb * 0.3 + score_peg * 0.3)
        
        return (score_pe * 0.6 + score_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE score: Higher is better
        score_roe = _linear_score(roe, 0.25, -0.10)
        # Margin score: Higher is better
        score_margin = _linear_score(margin, 0.20, -0.10)
        # Leverage score: Lower is better
        score_leverage = _linear_score(1.0 / (debt_equity + 0.01), 1.0, 3.0)

        return (score_roe * 0.4 + score_margin * 0.3 + score_leverage * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using ROE and PEG."""
        # Growth is often captured by the relationship between profit and price
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        
        # High ROE + Low PEG = Growth at a reasonable price
        score_roe = _linear_score(roe, 0.20, -0.05)
        # For growth, we want low PEG (high growth relative to PE)
        score_peg = _linear_score(1.0 / peg if peg > 0 else 2.0, 1.5, 0.5)

        return (score_roe * 0.5 + score_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
            
        return results