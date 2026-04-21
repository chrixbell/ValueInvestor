"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more balance with quality/growth
    "quality": 0.25,   # Increased to reward stable companies
    "growth": 0.25,    # Balanced with quality
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
            # We use a small epsilon to prevent zeroing out the entire score if one factor is 0,
            # but still allow for significant penalization.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean.
            # If product is 0, the result is 0.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB."""
        # Using common thresholds for A-share/HK markets. 
        # Low PE and low PB are prioritized.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # If PE is negative (loss making), we assign a low score. 
        # We use a fallback to avoid math errors with negative numbers in certain logic.
        # For value, we want low PE/PB.
        pe_score = 0.0
        if pe is not None and pe > 0:
            # PE range [0.5, 40] -> maps to [100, 0]
            pe_score = _linear_score(pe, best=0.5, worst=40.0)
        elif pe is not None and pe <= 0:
            # If PE is negative, it's usually a loss. We give it a very low score unless PB handles it.
            pe_score = 5.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            # PB range [0.5, 10] -> maps to [100, 0]
            pb_score = _linear_score(pb, best=0.5, worst=10.0)
        elif pb is not None and pb <= 0:
            pb_score = 5.0

        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        # ROE Score: Higher is better
        roe_score = 0.0
        if roe is not None:
            # Target ROE range [5%, 30%] -> maps to [0, 100]
            roe_score = _linear_score(roe, best=30.0, worst=5.0)
        
        # Debt/Equity Score: Lower is better
        de_score = 0.0
        if debt_equity is not None:
            # Target DE range [0, 1.5] -> maps to [100, 0]
            de_score = _linear_score(debt_equity, best=0.0, worst=1.5)
        elif debt_equity is None:
            de_score = 50.0

        # Weighted average of quality metrics
        return (roe_score * 0.7 + de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # ROE for growth (high ROE often indicates high growth potential/efficiency)
        roe_score = 0.0
        if roe is not None:
            roe_score = _linear_score(roe, best=25.0, worst=0.0)

        # PEG Score: Lower is better (Growth adjusted valuation)
        peg_score = 0.0
        if peg is not None and peg > 0:
            # PEG range [0.5, 2.5] -> maps to [100, 0]
            peg_score = _linear_score(peg, best=0.5, worst=2.5)
        elif peg is not None and peg <= 0:
            peg_score = 10.0

        return (roe_score * 0.5 + peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for result in results:
            self.score(result)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, result in enumerate(results):
            result.rank = i + 1
        return results