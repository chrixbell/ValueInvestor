"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced to allow more weight for quality and growth
    "quality": 0.30,   # Increased to prioritize stable companies
    "growth": 0.30,    # Balanced with quality for long-term returns
    "momentum": 0.00,  # Set to 0.0 as it's a fixed placeholder score
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
            # Add a small epsilon to avoid log(0) issues if using geometric logic, 
            # though here we use power-based product.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of (s/100)^w is already the composite scaled by 100^total_weight.
            # To get it back to [0, 100], we divide by 100^(total_weight - 1) or more simply:
            # (product of (s/100)^w) * 100 is not correct if total_weight != 1.
            # The standard way for weighted geometric mean is: exp( sum(w * ln(s)) / sum(w) )
            # We use a more stable approach:
            
            sum_log_scores = 0.0
            for k, score_val in weighted_scores_to_process.items():
                # Use a small epsilon to prevent log(0)
                s = max(1e-9, score_val / 100.0)
                sum_log_scores += self.weights[k] * math.log(s)
            
            composite_score = math.exp(sum_log_scores / total_weight) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score based on PE and PB."""
        # Using a blend of PE and PB. Lower is better.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle negative PE (common in loss-making stocks)
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Clamp PE between 1 and 40 for scoring
            pe_score = _linear_score(pe, 1.0, 40.0)
            # Invert: lower PE is better
            pe_score = 100.0 - pe_score
        elif pe is not None and pe <= 0:
            # If PE is negative, it's often a loss-maker. 
            # We give it a baseline score but not the best.
            pe_score = 20.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 0.1, 10.0)
            pb_score = 100.0 - pb_score
        elif pb is not None and pb <= 0:
            pb_score = 20.0

        # Weight PB and PE equally for the value sub-score
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score based on ROE and Debt."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity

        roe_score = 0.0
        if roe is not None:
            # ROE can be negative, but we treat it on a scale. 
            # Using linear interpolation for typical ROE range (-10% to 30%)
            roe_score = _linear_score(roe, -0.1, 0.3)
        
        debt_score = 50.0
        if debt_to_equity is not None:
            # Lower debt is better. Clamp between 0 and 1.5
            debt_score = _linear_score(debt_to_equity, 0.0, 1.5)
            debt_score = 100.0 - debt_score

        # Quality is driven heavily by ROE and solvency
        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # Growth score often correlates with ROE in value-investing contexts
        roe_score = 0.0
        if roe is not None:
            roe_score = _linear_score(roe, 0.05, 0.4)
        
        peg_score = 50.0
        if peg is not None and peg > 0:
            # Lower PEG is better (growth relative to value)
            peg_score = _linear_score(peg, 0.5, 3.0)
            peg_score = 100.0 - peg_score
        elif peg is None:
            peg_score = 50.0

        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results