"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow room for quality/growth
    "quality": 0.25,   # Increased to prioritize stable companies
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
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

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # We use a weighted arithmetic mean of normalized scores here to prevent 
        # a single zero sub-score from zeroing out the entire composite score, 
        # while still allowing high-quality factors to drive the rank.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a combination of PE and PB for valuation. 
        # If PE is missing, fallback to PB.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        if pe is not None and pe > 0:
            # Check if PB or PS provides a better context (not strictly necessary but helps)
            return _linear_score(pe, 5.0, 40.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 1.0, 5.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, 1.0, 10.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity

        score = 50.0
        # Quality is often driven by ROE.
        if roe is not None:
            roe_s = _linear_score(roe, 5.0, 25.0)
            score = score * 0.7 + roe_s * 0.3
        
        if debt_to_equity is not None:
            # Lower debt-to-equity is better (inverse linear)
            debt_s = _linear_score(1.0 / max(0.1, debt_to_equity), 0.1, 2.0)
            score = score * 0.7 + debt_s * 0.3
            
        return score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth potential (ROE + PEG context)."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        if roe is not None and peg is not None and peg > 0:
            # High ROE with low PEG is the sweet spot for growth investors
            # We combine them into a single metric: ROE / PEG (Earnings Yield/Growth hybrid)
            # But to keep it simple, let's score them separately and combine.
            roe_s = _linear_score(roe, 5.0, 30.0)
            peg_s = _linear_score(1/peg, 0.5, 3.0) # Higher PEG is worse
            return (roe_s + peg_s) / 2.0
        elif roe is not None:
            return _linear_score(roe, 5.0, 30.0)
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results