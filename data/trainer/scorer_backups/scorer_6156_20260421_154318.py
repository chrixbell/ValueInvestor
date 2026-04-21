"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more breathing room for quality/growth
    "quality": 0.25,   # Increased to emphasize stability
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Normalize score to [0.0, 1.0] range for geometric mean calculation
            # Using a small epsilon to prevent math errors if score_val is 0
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # We use a combination of PE/PB for valuation. 
        # If PE is negative (loss), we treat it as a high-risk/low value score.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Baseline: Use PE as primary, PB as secondary if PE is unavailable or extreme.
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range of 5 to 25.
            pe_score = _linear_score(pe, 5.0, 30.0)
            # Invert: lower PE is better for value
            pe_score = 100.0 - pe_score
        elif pe is not None and pe <= 0:
            pe_score = 10.0 # Low score for negative earnings

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.5, 5.0)
            pb_score = 100.0 - pb_score
        elif pb is not None and pb <= 0:
            pb_score = 10.0

        # Combine PE and PB scores
        if pe is not None and pb is not None:
            return (pe_score + pb_score) / 2.0
        return pe_score if pe is not None else pb_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        roe_score = 0.0
        if roe is not None:
            # High ROE is good. Scale around 15% target.
            roe_score = _linear_score(roe * 100, 5.0, 30.0)

        de_score = 0.0
        if debt_equity is not None:
            # Low debt-to-equity is good. 
            # We use a non-linear approach: high DE drops score quickly.
            de_score = _linear_score(debt_equity, 0.1, 2.0)
            de_score = 100.0 - de_score

        if roe is not None and debt_equity is not None:
            return (roe_score * 0.7) + (de_score * 0.3)
        return roe_score if roe is not None else de_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using PEG and Net Margin."""
        peg = result.valuation.peg_ratio
        margin = result.financials.net_margin

        peg_score = 0.0
        if peg is not None and peg > 0:
            # PEG < 1 is excellent.
            peg_score = _linear_score(peg, 0.5, 2.0)
            peg_score = 100.0 - peg_score
        elif peg is not None and peg <= 0:
            peg_score = 50.0 # Neutral if growth is negative/undefined

        margin_score = 0.0
        if margin is not None:
            # Higher margins indicate better growth-to-profitability.
            margin_score = _linear_score(margin * 100, 5.0, 25.0)

        if peg is not None and margin is not None:
            return (peg_score * 0.6) + (margin_score * 0.4)
        return peg_score if peg is not None else margin_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results