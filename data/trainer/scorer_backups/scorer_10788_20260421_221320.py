"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more balance
    "quality": 0.25,   # Increased quality weight
    "growth": 0.25,    # Increased growth weight
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
            # To prevent a zero score from wiping out the entire product (geometric mean),
            # we add a tiny epsilon to the normalized score. 
            # However, if we want to penalize bad stocks heavily, we use a small offset.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean.
            # Since we used (score/100)^weight, the product is already scaled.
            # We divide by total_weight in the exponent to normalize weights if they don't sum to 1.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Evaluate valuation metrics."""
        # Use a combination of PE and PB for value.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dividend = result.valuation.dividend_yield

        # Fallback values to avoid division by zero or None issues
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        div = dividend if dividend is not None else 2.0

        # Value score: lower PE and PB are better, higher dividend is better.
        # We use a simple weighted sum of normalized scores for sub-factors.
        val_pe = _linear_score(pe, 5.0, 40.0) # Lower is better (inverted in linear_score logic if we swap best/worst?)
        # Wait, _linear_score: (value - worst) / (best - worst). 
        # For PE, 'best' is 5, 'worst' is 40. If pe=5, score = (5-40)/(5-40) = 1. Correct.
        # Let's re-implement logic for "lower is better" properly within the scoring context.
        
        # Re-mapping: best=low_pe, worst=high_pe
        score_pe = _linear_score(pe, 5.0, 40.0) if pe is not None else 50.0
        # Actually, the current _linear_score: if value=5, best=5, worst=40 -> (5-40)/(5-40) = 1.0 * 100 = 100.
        # If value=40, best=5, worst=40 -> (40-40)/(5-40) = 0. Correct.
        
        # Let's use a more robust approach for the sub-scores:
        # We want to map input values to [0, 100] where 100 is best.
        
        # PE score (lower is better)
        if pe is not None and pe > 0:
            # If PE is 5 (best), score 100. If PE is 40 (worst), score 0.
            s_pe = max(0.0, min(100.0, (40.0 - pe) / (40.0 - 5.0) * 100.0)) if pe < 40.0 else 0.0
            # Wait, the logic above is: if pe=5 -> (40-5)/35 * 100 = 100. If pe=40 -> 0.
            # Let's use a simpler clamping:
            if pe <= 5.0: s_pe = 100.0
            elif pe >= 40.0: s_pe = 0.0
            else: s_pe = (40.0 - pe) / 35.0 * 100.0
        else:
            s_pe = 50.0

        if pb is not None and pb > 0:
            if pb <= 1.0: s_pb = 100.0
            elif pb >= 5.0: s_pb = 0.0
            else: s_pb = (5.0 - pb) / 4.0 * 100.0
        else:
            s_pb = 50.0

        if dividend is not None:
            # High dividend is good. 0% -> 0, 5% -> 100
            s_div = max(0.0, min(100.0, dividend * 20.0))
        else:
            s_div = 50.0

        return (s_pe * 0.4 + s_pb * 0.3 + s_div * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Evaluate quality metrics (ROE, Debt/Equity)."""
        roe = result.financials.roe
        debt_eq = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE score (higher is better)
        if roe is not None:
            s_roe = max(0.0, min(100.0, (roe + 0.2) * 100.0)) # Assuming ROE is decimal (e.g. 0.15)
            # Wait, if ROE is 0.15 (15%), score = 35? No, let's assume ROE is like 15.0
            if abs(roe) < 0.001: s_roe = 50.0
            else:
                # Scale ROE: 20% is 100, -10% is 0
                s_roe = max(0.0, min(100.0, (roe - (-0.1)) / (0.2 - (-0.1)) * 100.0))
        else:
            s_roe = 50.0

        # Debt/Equity (lower is better)
        if debt_eq is not None:
            # 0 debt = 100, 2.0 debt = 0
            s_debt = max(0.0, min(100.0, (2.0 - debt_eq) / 2.0 * 100.0))
        else:
            s_debt = 50.0

        # Margin (higher is better)
        if margin is not None:
            s_margin = max(0.0, min(100.0, (margin - (-0.05)) / (0.3 - (-0.05)) * 100.0))
        else:
            s_margin = 50.0

        return (s_roe * 0.4 + s_debt * 0.3 + s_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Evaluate growth metrics (PEG)."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG (lower is better)
        if peg is not None and peg > 0:
            s_peg = max(0.0, min(100.0, (3.0 - peg) / 2.5 * 100.0))
        else:
            s_peg = 50.0

        # ROE as growth proxy (higher is better)
        if roe is not None:
            s_roe = max(0.0, min(100.0, (roe - (-0.1)) / 0.3 * 100.0))
        else:
            s_roe = 50.0

        return (s_peg * 0.6 + s_roe * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)

        for i, r in enumerate(results):
            r.rank = i + 1
        return results