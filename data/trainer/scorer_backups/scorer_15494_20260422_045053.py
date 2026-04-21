"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Lowered slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to emphasize stability and profitability
    "growth": 0.25,    # Balanced with quality to capture growth-at-reasonable-price
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

        total_weight = sum(self.weights[k] for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        # We use a slightly modified geometric mean approach. 
        # To prevent a single zero score from wiping everything out (which can happen with pure geometric),
        # but still maintaining the penalty for poor fundamentals, we use a small epsilon-based approach.
        # However, to keep it clean and robust for the current structure:
        product_of_powers = 1.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            # Normalize score to [0, 1]. We add a tiny epsilon to prevent log(0) issues in math
            # but since we are multiplying, 0 is fine.
            normalized_score = score_val / 100.0
            product_of_powers *= (normalized_score ** weight)

        # The product_of_powers is already (S1/100)^w1 * (S2/100)^w2...
        # To get the final score back to [0, 100], we don't need to take the root if weights sum to 1.
        # If they don't sum to 1, we adjust:
        # (S_normalized) = product^(1/total_weight)
        
        if total_weight > 0:
            # Use the product of powers directly as it represents the weighted geometric mean scaled by 100^total_weight
            # To normalize back to [0, 100] regardless of total_weight:
            # We need (product) ^ (1 / total_weight) * 100
            # But wait, the math: (S1^w1 * S2^w2) where weights are like 0.5, 0.5. Product is in [0,1].
            # If total_weight is 1.0, result is in [0,1].
            # If total_weight is 2.0, result is in [0,1].
            # So we take the root to normalize.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PE if available, else PB.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        if pe is not None and pe > 0:
            # PE-based scoring. Clamp to reasonable bounds for Chinese market.
            return _linear_score(pe, 40.0, 2.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 5.0, 0.5)
        elif result.valuation.dividend_yield is not None:
            return _linear_score(result.valuation.dividend_yield, 10.0, 0.0)
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # Base score from ROE
        if roe is not None:
            # ROE can be negative, so we use a logic that handles it. 
            # For simplicity in linear_score, we assume ROE is positive or handled by bounds.
            # Using a range of -20% to 40%.
            q_score = _linear_score(roe, 25.0, -10.0)
        else:
            q_score = 50.0

        # Adjust for leverage (Debt/Equity)
        if debt_equity is not None:
            # High debt reduces quality. 
            # If debt/equity is high, it subtracts from the score.
            leverage_penalty = _linear_score(debt_equity, 2.0, 0.5) # Higher debt = lower score
            # We blend them: if we have ROE, we use it as primary.
            if roe is not None:
                # We want high ROE and low leverage. 
                # If debt_equity is very high, it should pull the score down.
                if debt_equity > 2.0: # Very high leverage
                    q_score *= 0.5
                elif debt_equity < 0.5: # Very low leverage
                    q_score = min(100.0, q_score * 1.2)
            else:
                q_score = leverage_penalty

        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        if peg is not None and peg > 0:
            # PEG-based scoring (Lower is better)
            return _linear_score(peg, 1.0, 3.0)
        elif roe is not None:
            # Fallback to ROE-based growth score
            return _linear_score(roe, 20.0, -5.0)
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results