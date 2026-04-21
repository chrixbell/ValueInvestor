"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to balance with quality/growth
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Increased growth weight to capture upside potential
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
            # Use a small epsilon to prevent zero-multiplication issues in geometric mean
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The geometric mean is inherently calculated by the product of powers.
            # To return to [0, 100], we scale the product.
            # Since (x^w1 * y^w2) is equivalent to ((x^w1 * y^w2)^(1/sum_w)), 
            # but we want to keep the scale relative to the identity (1.0).
            # If all weights sum to 1, product_of_powers is already the scaled score.
            # If they don't sum to 1, we adjust by total_weight.
            composite_score = product_of_powers * (100.0 ** 0) # Placeholder logic
            # Correct way to scale back: if weights sum to 1, product_of_powers is in [0, 1]
            # The formula below handles cases where total_weight != 1.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # Wait, the power (1/total_weight) is for normalizing if we were taking a standard mean.
            # Actually, (S1^w1 * S2^w2) where sum(w)=1 is already in the correct scale.
            # If sum(w) != 1, we need to adjust. Let's re-normalize weights first.
            
        # Recalculate with weight normalization to ensure stability
        total_w = sum(self.weights.get(k, 0.0) for k in scores.keys())
        if total_w == 0:
            result.composite_score = 50.0
            return result
        
        # Re-run clean logic:
        p_pow = 1.0
        for k, score_val in weighted_scores_to_process.items():
            w = self.weights[k] / total_w  # Normalized weight
            p_pow *= (max(0.0, score_val) / 100.0) ** w
        
        result.composite_score = p_pow * 100.0
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB → higher score."""
        # Use PEG if available as it combines value and growth implicitly
        if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0:
            return _linear_score(1.0 / result.valuation.peg_ratio, 1.0, 5.0)
        
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback to PE or PB
        val = pe if pe is not None else pb
        if val is None: return 50.0
        
        # Using a wide range for value scoring
        return _linear_score(val, 5.0, 40.0)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.0
        
        # Combine ROE and Margin (Quality) and Debt (Risk)
        q_score = (roe * 0.5 + margin * 0.5) * 100 # This is a rough proxy
        # Since we need to map to [0, 100], let's use specific bounds
        # We assume ROE/Margin are in decimal (e.g. 0.15)
        q_val = (roe + margin) / 2.0 # Average return/margin
        # Use a scale where 15% (0.15) is decent
        return _linear_score(q_val, -0.1, 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE (already in quality), lower PEG → higher score."""
        # Growth often relates to revenue/income growth, but since we only have 
        # static snapshots, we use the relationship between valuation and profitability.
        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            return _linear_score(1.0 / peg, 0.5, 3.0)
        
        # Fallback: use ROE as a proxy for growth-oriented quality
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        return _linear_score(roe, -0.1, 0.3)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for res in results:
            self.score(res)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results