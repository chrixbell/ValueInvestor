"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced value weight to allow more room for quality/growth
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Increased growth weight to capture upside potential
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
    """Return a score in [0, 100] via logarithmic interpolation."""
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

    def score(self, result: Screening_Result) -> Screening_Result:
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
            # Use a small epsilon to prevent zero-score issues in geometric mean if needed, 
            # but here we allow 0 to propagate as per previous logic.
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            # The formula (product_of_powers) already accounts for weight scaling.
            # Since product_of_powers = PI( (score/100)^weight ), 
            # the result is already normalized if weights sum to 1.
            # We use a slightly more robust geometric mean approach:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Use PE if available, else PB. 
        # We want low PE/PB to be high score.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Fallback/Default logic
        if pe is not None and pe > 0:
            # Using a range of 0 to 30 for PE as 'worst' (low score) and 5 as 'best'.
            # This captures value in reasonable ranges.
            return _linear_score(pe, 5.0, 30.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 1.0, 5.0)
        elif result.valuation.dividend_yield is not None:
            # If no PE/PB, use dividend yield (higher is better)
            return _linear_score(result.valuation.dividend_yield, 5.0, 0.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE is a primary quality metric
        if roe is not None:
            # Map reasonable ROE (e.g., 2% to 25%) to score
            # We use a custom approach: higher ROE is better. 
            # But we need to handle the 'worst' being a low ROE.
            # Let's use a simple linear scaling for ROE.
            roe_score = _linear_score(roe, 20.0, -5.0)
            
            if debt_equity is not None:
                # Scale debt/equity (lower is better)
                de_score = _linear_score(debt_equity, 0.5, 2.0)
                # Combine ROE and Debt/Equity (weighted 70/30)
                return (roe_score * 0.7) + (de_score * 0.3)
            return roe_score
        
        # Fallback to Current Ratio if ROE is missing
        current_ratio = result.financials.current_ratio
        if current_ratio is not None:
            return _linear_score(current_ratio, 2.0, 0.5)
            
        return 50.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using PEG and margin stability."""
        peg = result.valuation.peg_ratio
        gross_margin = result.financials.gross_margin

        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, lower is better (growth at reasonable price)
            # We map PEG 0.5 -> 100, 3.0 -> 0
            return _linear_score(peg, 0.5, 3.0)
        elif gross_margin is not None:
            # Use gross margin as a proxy for growth/moat if PEG is unavailable
            return _linear_score(gross_margin, 40.0, 10.0)
        
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        # Sort descending (highest score first)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results