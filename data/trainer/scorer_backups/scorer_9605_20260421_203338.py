"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture fundamental stability
    "growth": 0.25,    # Increased to capture upside potential
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
            # Small epsilon added to prevent zero-score annihilation in geometric mean
            # while still allowing low scores to pull the composite down.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers is already weighted. We just need to scale it back.
            # Since (S1^w1 * S2^w2) where sum(w)=1 is the geometric mean, 
            # and we are using normalized scores (0-1), we scale by 100.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Prioritize Forward PE if available, then trailing PE.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Use a fallback for negative/None PE
        if pe is None or pe <= 0:
            pe_score = 50.0
        else:
            # Clamp PE between 1 and 40 for scoring purposes.
            pe_score = _linear_score(pe, 1.0, 40.0)

        if pb is None or pb <= 0:
            pb_score = 50.0
        else:
            # Clamp PB between 0.1 and 15 for scoring purposes.
            pb_score = _linear_score(pb, 0.1, 15.0)

        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.0

        # ROE: Higher is better. Target range [0% to 30%]
        roe_score = _linear_score(roe * 100, 0.0, 30.0)

        # Debt to Equity: Lower is better. Target range [0% to 100%]
        # If debt_to_equity is 0, score is 100. If 100 (1.0), score is 0.
        if debt_to_equity < 0: # Handle edge cases
            debt_score = 50.0
        else:
            debt_score = _linear_score(debt_to_equity * 100, 0.0, 100.0)
            debt_score = 100.0 - debt_score

        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and gross margin."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        gross_margin = result.financials.gross_margin if result.financials.gross_margin is not None else 0.0

        # PEG: Lower is better (growth at a reasonable price).
        if peg is None or peg <= 0:
            peg_score = 50.0
        else:
            # A PEG of 0.5 is great (100), a PEG of 3.0 is poor (0).
            peg_score = _linear_score(peg, 0.5, 3.0)

        # Gross Margin: Higher is better. Target range [0% to 50%]
        gm_score = _linear_score(gross_margin * 100, 0.0, 50.0)

        return (peg_score * 0.5) + (gm_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
            
        return results