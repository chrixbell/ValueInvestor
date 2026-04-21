"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more room for quality/growth
    "quality": 0.25,   # Increased to reward stable business models
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
            # Normalize score to [0, 1] for geometric mean. 
            # We use a small epsilon (1e-6) to prevent log(0) issues if using geometric 
            # logic, though here we just multiply. A score of 0 will zero out the product.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores scaled back to 100.
            # Since product_of_powers is (s1^w1 * s2^w2...), and sum(wi) might not be 1,
            # we use the standard formula: (product)^(1/sum_weights)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score using PE and PB ratios."""
        # Use pe_forward if available, otherwise pe_ratio. 
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle cases where PE/PB are None or non-positive
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Target PE range [1, 20] for high score
            pe_score = _linear_score(pe, 1.0, 25.0)
        
        pb_score = 0.0
        if pb is not None and pb > 0:
            # Target PB range [0.5, 3] for high score
            pb_score = _linear_score(pb, 0.5, 5.0)

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score using ROE and debt-to-equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE scoring (higher is better)
        roe_score = _linear_score(roe, 15.0, 30.0) if roe != 0 else 0.0
        if roe > 30: roe_score = 100.0
        elif roe < -5: roe_score = 0.0

        # Debt-to-equity scoring (lower is better)
        # If debt_to_equity is 0 or negative (unlikely), it's a perfect score
        if debt_to_equity <= 0:
            debt_score = 100.0
        else:
            # Target debt-to-equity range [0, 1]
            debt_score = _linear_score(debt_to_equity, 0.2, 1.5)

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Computes growth score using PEG and gross margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        gross_margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0

        # PEG score (lower is better)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.0)
        elif peg is not None and peg <= 0:
            # Negative PEG implies high growth relative to PE, but can be risky.
            peg_score = 100.0 if peg < 0 else 50.0
        else:
            peg_score = 50.0

        # Gross Margin score (higher is better)
        gm_score = _linear_score(gross_margin, 20.0, 50.0) if gross_margin is not None else 50.0
        if gross_margin is not None and gross_margin > 50: gm_score = 100.0
        elif gross_margin is not None and gross_margin < 0: gm_score = 0.0

        return (peg_score + gm_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks the results based on composite score."""
        for res in results:
            self.score(res)

        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)

        for i, res in enumerate(results):
            res.rank = i + 1
        return results