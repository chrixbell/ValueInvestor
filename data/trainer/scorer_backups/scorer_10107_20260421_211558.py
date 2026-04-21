"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Adjusted to balance with quality/growth
    "quality": 0.25,   # Increased to reward stable businesses
    "growth": 0.25,    # Balanced with quality
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
            # We use a small epsilon to avoid issues with log(0) in geometric mean 
            # if score_val is exactly 0.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean.
            # Since we used normalized_score = score/100, the result is in [0, 1]
            # We apply (1/total_weight) to ensure the exponentiation doesn't distort 
            # if weights don't sum to 1.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluate value based on PE and PB ratios."""
        # Use PEG as a secondary check if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Heuristic thresholds for "good" and "bad" value
        # PE: 5-30 is reasonable, <5 is very cheap, >40 is expensive
        # PB: 0.5-3 is reasonable, <0.5 is very cheap, >10 is expensive
        
        v_score = 50.0
        
        # Handle PE (Avoid division by zero or negative PE issues)
        if pe is not None and pe > 0:
            # Map PE 5 -> 100, PE 40 -> 0
            v_score = _linear_score(pe, 5.0, 40.0)
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; often indicates loss. Score low but not zero.
            v_score = 10.0
        else:
            # Fallback to PB if PE is missing
            if pb is not None and pb > 0:
                v_score = _linear_score(pb, 0.5, 5.0)
            else:
                v_score = 50.0

        # If PEG is available, it's often a better value metric
        if peg is not None and peg > 0:
            # PEG 0.5 -> 100, PEG 3.0 -> 0
            peg_score = _linear_score(peg, 0.5, 3.0)
            # Blend PE/PB score with PEG
            v_score = (v_score * 0.4) + (peg_score * 0.6)

        return v_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality via ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        q_score = 50.0
        roe_part = 50.0
        de_part = 50.0

        # ROE is a primary quality metric
        if roe is not None:
            # Assume 20% ROE is great, 5% is poor
            roe_part = _linear_score(roe * 100, 5.0, 25.0)
        
        # Debt-to-equity: lower is better for quality
        if debt_equity is not None:
            # Assume 0.2 is great, 1.5 is poor
            de_part = _linear_score(debt_equity, 0.2, 1.5)
        elif debt_equity is None:
            # If no data, assume neutral if ROE exists, otherwise 50
            de_part = 50.0

        # Combine: ROE is usually more important for quality
        q_score = (roe_part * 0.7) + (de_part * 0.3)
        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth via ROE and margin stability."""
        # In this simple model, we use ROE as a proxy for growth potential 
        # but can also look at revenue/net income if we had historicals.
        # Since we only have current snapshot, we use margin and ROE.
        roe = result.financials.roe
        margin = result.financials.net_margin

        g_score = 50.0
        if roe is not None and margin is not None:
            # High ROE + High Margin = Strong growth capability/efficiency
            # Scale 0-100 based on combined efficiency
            g_score = (roe * 100 + margin * 100) / 2.0
            # Clamp to reasonable bounds for growth-specific score
            g_score = _linear_score(g_score, 5.0, 30.0)
        elif roe is not None:
            g_score = _linear_score(roe * 100, 5.0, 30.0)
        elif margin is not None:
            g_score = _linear_score(margin * 100, 5.0, 30.0)
            
        return g_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results