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
    "quality": 0.25,   # Increased quality weight to capture more stable returns
    "growth": 0.25,    # Increased growth weight to capture upside potential
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

        # Use a small epsilon to prevent math errors with geometric mean if score is 0
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] for geometric mean. 
            # We add a tiny epsilon so that a 0 score (bad factor) doesn't zero out the whole composite,
            # but still heavily penalizes it.
            normalized_score = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculates value score using PE and PB ratios."""
        # Using a combination of PE and PB for valuation. 
        # We use log_score to handle the wide distribution of ratios.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Default values for missing data to prevent failure
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0

        # Score 1: PE (Lower is better)
        # We want to score low PE high. Since log_score assumes higher is better, 
        # we use the inverse logic by using a target range.
        # For PE, 5 is 'best', 40 is 'worst'.
        pe_score = _linear_score(1.0/pe, 1.0/40.0, 1.0/5.0) if pe > 0 else 0.0
        pb_score = _linear_score(1.0/pb, 1.0/10.0, 1.0/1.0) if pb > 0 else 0.0
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if (result.financials.debt_to_equity is not None and result.financials.debt_to_equity > 0) else 1.0
        
        # ROE is a primary quality metric (higher is better)
        roe_score = _linear_score(roe, 30.0, -10.0)
        # Debt/Equity (lower is better). We use the reciprocal-like logic.
        de_score = _linear_score(1.0/debt_equity, 1.0/3.0, 1.0/0.5)
        
        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if (result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0) else 5.0
        
        # ROE as a proxy for internal growth capability
        roe_score = _linear_score(roe, 20.0, -5.0)
        # PEG (Lower is better). High PEG means growth is too expensive.
        peg_score = _linear_score(1.0/peg, 1.0/5.0, 1.0/0.5)
        
        return (roe_score + peg_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks the results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results