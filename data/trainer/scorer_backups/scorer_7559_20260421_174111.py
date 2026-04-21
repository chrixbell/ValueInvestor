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
    "quality": 0.25,   # Increased weight to favor stable companies
    "growth": 0.25,    # Balanced with quality for a more robust profile
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
            # Using a small epsilon to prevent zero-multiplication issues in geometric mean
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers (S1^w1 * S2^w2...) is already normalized by the weights
            # because we want (S1^w1 * S2^w2...)^(1/total_weight).
            # We simplify the math:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use trailing PE as primary, fallback to PB or forward PE
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        f_pe = result.valuation.pe_forward

        # Handle negative PE (loss making) by treating it as high risk/low value
        if pe is None or pe <= 0:
            pe = 100.0 if pb is None else (pb * 10.0) # Heuristic fallback
        if pe <= 0: pe = 1.0

        # Thresholds for linear scoring
        pe_best, pe_worst = 10.0, 40.0
        pb_best, pb_worst = 1.0, 5.0

        # Calculate component scores
        # If PE is very low, it's good. If PB is very low, it's good.
        v_pe = _linear_score(pe, pe_best, pe_worst) if pe > 0 else 0.0
        v_pb = _linear_score(pb, pb_best, pb_worst) if (pb and pb > 0) else 0.0
        
        # If pe is extremely low (e.g. 2), linear_score might clamp it to 100.
        # If pe is negative, we need a way to score it. 
        # Let's refine: if pe is negative, it's not necessarily "bad" (could be high growth), 
        # but for a value scorer, we treat it as 0.
        
        return (v_pe + v_pb) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE score: higher is better
        roe_val = roe if roe is not None else 0.0
        # Using a range of -0.2 to 0.4 for ROE
        q_roe = _linear_score(roe_val, 0.25, -0.1)

        # Leverage score: lower is better
        de_val = debt_equity if debt_equity is not None else 1.0
        q_leverage = _linear_score(de_val, 0.5, 2.0)

        return (q_roe + q_leverage) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # PEG score: lower is better (growth at reasonable price)
        # If PEG is None, we skip it or use a neutral value
        if peg is not None and peg > 0:
            g_peg = _linear_score(peg, 0.5, 3.0)
        else:
            g_peg = 50.0

        # ROE as a proxy for growth/efficiency
        roe_val = roe if roe is not None else 0.0
        g_roe = _linear_score(roe_val, 0.2, -0.1)

        return (g_peg + g_roe) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sorts results by composite score and assigns ranks."""
        # Sort descending: highest score first
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results