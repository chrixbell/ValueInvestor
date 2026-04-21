"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced value weight slightly to allow more room for quality/growth
    "quality": 0.25,   # Increased quality to reward stable businesses
    "growth": 0.25,    # Balanced growth with quality
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

        # We use a weighted arithmetic mean of normalized scores for stability.
        # A geometric mean can be overly sensitive to a single zero score (e.g., if one factor is 0).
        # For high-dimensional stock scoring, arithmetic mean of normalized scores is more robust.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # We use the score directly as it is already in [0, 100]
            total_weighted_sum += (score_val * weight)

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        v = result.valuation
        # Use a mixture of PE and PB for valuation. 
        # If PE is missing, fallback to PB or PS.
        if v.pe_ratio and v.pe_ratio > 0:
            # Using a broad range for PE (1 to 40)
            score = _linear_score(v.pe_ratio, 1.0, 40.0)
            # Invert: low PE is good
            return 100.0 - score
        elif v.pb_ratio and v.pb_ratio > 0:
            score = _linear_score(v.pb_ratio, 1.0, 15.0)
            return 100.0 - score
        elif v.ps_ratio and v.ps_ratio > 0:
            score = _linear_score(v.ps_ratio, 0.1, 5.0)
            return 100.0 - score
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        f = result.financials
        q_score = 0.0
        count = 0

        # ROE is a primary quality metric
        if f.roe:
            q_score += _linear_score(f.roe, -10.0, 30.0)
            count += 1
        
        # Debt to Equity (Lower is better)
        if f.debt_to_equity is not None:
            # Clamp debt to equity between 0 and 2.0 for scoring logic
            d_score = _linear_score(f.debt_to_equity, 0.0, 2.0)
            q_score += (100.0 - d_score)
            count += 1

        # Gross Margin
        if f.gross_margin is not None:
            gm_score = _linear_score(f.gross_margin, 0.0, 50.0)
            q_score += gm_score
            count += 1

        if count == 0:
            return 50.0
        return q_score / count

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        v = result.valuation
        f = result.financials
        g_score = 0.0
        count = 0

        # PEG Ratio (Lower is better)
        if v.peg_ratio and v.peg_ratio > 0:
            score = _linear_score(v.peg_ratio, 0.1, 3.0)
            g_score += (100.0 - score)
            count += 1
        
        # ROE (Growth proxy)
        if f.roe:
            g_score += _linear_score(f.roe, 0.0, 25.0)
            count += 1

        if count == 0:
            return 50.0
        return g_score / count

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite_score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite_score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results