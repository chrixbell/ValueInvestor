"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced value to allow more room for quality/growth
    "quality": 0.25,   # Increased quality to capture stable earners
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # However, dividing by 100 is standard for the range [0, 1]
            normalized_val = score_val / 100.0
            # Clamp to a tiny positive value to ensure stability in geometric mean calculation
            safe_val = max(1e-6, normalized_val)
            product_of_powers *= (safe_val ** weight)

        if total_weight > 0:
            # The geometric mean scaling is (product)^(1/total_weight)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE if available, otherwise Trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle potential None or zero/negative PE
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Typical healthy PE range for value stocks: 5 to 25
            pe_score = _linear_score(pe, 5.0, 25.0)
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; if it's extremely negative, it might be a value trap or highly profitable
            # We treat non-positive PE as high score (potential deep value) but clamp via the linear function
            pe_score = 100.0 if pe < 0 else 50.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            # Typical healthy PB range: 0.5 to 4.0
            pb_score = _linear_score(pb, 0.5, 4.0)
        elif pb is not None and pb <= 0:
            pb_score = 100.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE Score: Higher is better (target range: 5% to 20%)
        # We use linear mapping. Note that ROE is often expressed as decimal (0.15) or percentage (15.0)
        # Assuming input is decimal-based but adjusted for common ranges.
        roe_val = roe * 100 if abs(roe) < 2 else roe
        roe_score = _linear_score(roe_val, 5.0, 25.0)

        # Debt-to-Equity Score: Lower is better
        # If debt_to_equity is 0.5 (50%), it's good. If 2.0, it's risky.
        # We invert the logic: best is low debt (e.g., 0), worst is high (e.g., 2.0)
        if debt_to_equity < 0: # Handle edge case of negative equity
            de_score = 100.0
        else:
            # Map 0.0 -> 100 and 2.0 -> 0
            de_score = _linear_score(debt_to_equity, 0.0, 2.0)

        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0
        
        # Margin adjustment (if expressed as decimal)
        margin_val = margin * 100 if abs(margin) < 2 else margin

        # PEG Score: Lower is better (ideal around 1.0)
        if peg is not None and peg > 0:
            # Map PEG 0.5 -> 100, PEG 3.0 -> 0
            peg_score = _linear_score(peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            # Negative PEG (negative earnings) - usually high risk/growth potential
            peg_score = 50.0
        else:
            peg_score = 50.0

        # Margin Score: Higher is better (target range: 15% to 40%)
        margin_score = _linear_score(margin_val, 15.0, 40.0)

        return (peg_score + margin_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
            
        return results