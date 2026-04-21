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
    "quality": 0.30,   # Increased to capture fundamental stability
    "growth": 0.20,    # Retained growth component
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
            # Use a small epsilon to prevent log(0) issues in geometric mean logic
            # but allow 0 to be a valid score.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores (S/100) scaled back to 100
            # product_of_powers is (S1/100)^w1 * ...
            # We divide by total_weight if we want the weighted geometric mean root.
            # However, to keep it in [0, 100], we use the product directly.
            # If weights sum to 1, product_of_powers is already the scale-adjusted result.
            # To handle weights != 1, we adjust:
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None/Zero cases for PE
        pe_val = pe if (pe is not None and pe > 0) else 20.0
        pb_val = pb if (pb is not None and pb > 0) else 2.0

        # Score for PE (Lower is better)
        # We use a range where 5 is best, 30 is worst.
        pe_s = _linear_score(pe_val, 5.0, 30.0)
        # Score for PB (Lower is better)
        pb_s = _linear_score(pb_val, 1.0, 5.0)

        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better (Targeting 15-20% as best)
        roe_s = _linear_score(roe, 20.0, 0.0)
        # Debt/Equity: Lower is better (Targeting < 0.5 as best)
        de_s = _linear_score(debt_equity, 0.2, 1.5)

        return (roe_s + de_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG: Lower is better (Targeting 1.0 as best)
        peg_s = _linear_score(peg, 0.5, 3.0)
        # ROE as a proxy for growth potential/efficiency
        roe_s = _linear_score(roe, 15.0, -5.0)

        return (peg_s + roe_s) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results