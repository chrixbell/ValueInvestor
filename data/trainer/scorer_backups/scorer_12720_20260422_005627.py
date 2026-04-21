"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Rebalanced: Value is important but shouldn't dominate quality/growth entirely.
    "quality": 0.25,   # Increased: Quality is a strong predictor of long-term returns and stability.
    "growth": 0.25,    # Increased: Growth complements quality in a multi-factor approach.
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score.
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

        # Adding a small epsilon to avoid math errors with zero scores in geometric mean
        epsilon = 1e-6

        total_weight = 0.0
        product_of_powers = 1.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use (score + epsilon) to ensure non-zero values for geometric mean calculation
            product_of_powers *= ((score_val + epsilon) / 100.0) ** weight

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # We use a combination of PE and PB. 
        # For stocks with no PE, we rely on PB (and vice versa).
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Define reasonable bounds for linear scoring
        pe_best, pe_worst = 10.0, 50.0
        pb_best, pb_worst = 1.0, 5.0

        v_score = 0.0
        count = 0

        if pe is not None and pe > 0:
            # Lower PE is better for value
            v_score += (1.0 - (pe - pe_worst) / (pe_best - pe_worst)) * 100.0 if pe_best != pe_worst else 50.0
            count += 1
        elif pe is not None and pe <= 0:
            v_score += 100.0 # Negative PE is often high growth/turnaround
            count += 1

        if pb is not None and pb > 0:
            v_score += (1.0 - (pb - pb_worst) / (pb_best - pb_worst)) * 100.0 if pb_best != pb_worst else 50.0
            count += 1

        if count == 0:
            return 50.0
        
        # Clamp and average the value components
        final_v = v_score / count
        return max(0.0, min(100.0, final_v))

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        d_e = result.financials.debt_to_equity

        q_score = 0.0
        count = 0

        if roe is not None:
            # Higher ROE is better. Using a log scale for growth/quality capture.
            # We treat negative ROE as 0.
            val = max(0.0, roe)
            q_score += _log_score(val + 0.01, 30.0, 0.01)
            count += 1

        if d_e is not None:
            # Lower Debt-to-Equity is better.
            # Clamp D/E to a reasonable range [0, 2] for scoring.
            clamped_de = max(0.0, min(2.0, d_e))
            q_score += (1.0 - clamped_de / 2.0) * 100.0
            count += 1

        if count == 0:
            return 50.0
        
        return max(0.0, min(100.0, q_score / count))

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio
        gm = result.financials.gross_margin

        g_score = 0.0
        count = 0

        if peg is not None and peg > 0:
            # Lower PEG is better (Growth relative to value)
            # Mapping [0, 3] roughly to [100, 0]
            val = max(0.0, min(3.0, peg))
            g_score += (1.0 - val / 3.0) * 100.0
            count += 1

        if gm is not None:
            # Higher Gross Margin is better.
            val = max(0.0, gm)
            g_score += _log_score(val + 0.01, 50.0, 0.01)
            count += 1

        if count == 0:
            return 50.0
        
        return max(0.0, min(100.0, g_score / count))

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)

        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)

        for i, res in enumerate(sorted_results):
            res.rank = i + 1

        return sorted_results