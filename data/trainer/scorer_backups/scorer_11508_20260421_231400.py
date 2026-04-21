"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced slightly to allow more breathing room for quality/growth
    "quality": 0.25,   # Increased to prioritize stability and profitability
    "growth": 0.30,    # Increased to capture expansionary potential
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
            # Use a small epsilon to prevent zero issues in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            # If score is 0, we use a tiny epsilon to allow the weight to exist without zeroing everything
            # but since it's a product, 0 will still result in 0. This is intended for "bad" stocks.
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers (S1^w1 * S2^w2...) already accounts for the weights.
            # To normalize it back to [0, 100], we adjust by the total weight.
            # However, if sum(weights) != 1, we need to normalize the exponent.
            # The math: (S_normalized ^ weight) -> if sum(weights) is 1, this is the geometric mean.
            # If not, we want (Product)^(1/sum_weights).
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Use Forward PE if available, else trailing PE. 
        # If both None, use PB.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Thresholds for clamping (industry standard-ish)
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        score = 0.0
        count = 0

        if pe is not None and pe > 0:
            # Lower PE is better for value
            score += _linear_score(pe, pe_best, pe_worst)
            count += 1
        
        if pb is not None and pb > 0:
            # Lower PB is better for value
            score += _linear_score(pb, pb_best, pb_worst)
            count += 1

        if count == 0:
            return 50.0
        return score / count

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE: Higher is better (Targeting ~15-25%)
        roe_score = _linear_score(roe * 100, 25.0, 0.0)
        
        # Debt/Equity: Lower is better (Targeting < 1.0)
        # We use a simple linear scale: 0 debt is best, 2.0+ is worst
        de_score = _linear_score(debt_equity, 0.2, 1.5)
        
        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using PEG and Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.1
        
        # PEG: Lower is better (Targeting 1.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 1.0, 3.0)
        else:
            # If no PEG, fallback to a neutral-high value if it's likely growing
            peg_score = 50.0

        # Margin: Higher is better (Stability of growth)
        margin_score = _linear_score(margin * 100, 20.0, 5.0)

        return (peg_score + margin_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)

        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign ranks (1-based)
        for i, res in enumerate(results):
            res.rank = i + 1
        
        return results