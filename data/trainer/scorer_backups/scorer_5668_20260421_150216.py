"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Focus on value-driven stability
    "quality": 0.25,   # Increase quality weight to capture robust businesses
    "growth": 0.25,    # Balanced with quality for long-term returns
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

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Use an arithmetic mean of normalized scores to prevent a single 0 score
        # from zeroing out the entire composite (as happens in geometric mean).
        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                total_weighted_score += score_val * weight
                total_weight += weight

        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score based on PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Handle None/Zero cases for PE
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Assume a healthy PE range for scoring
            pe_score = _linear_score(15.0, 5.0, 30.0) if pe > 0 else 0.0
            # Actually, let's use a more robust inversion:
            pe_score = _linear_score(1.0/pe, 1.0/30.0, 1.0/5.0) if pe > 0 else 0.0
        
        # Re-implementing logic: lower is better for PE/PB
        # We'll use a simple mapping logic
        pe_val = pe if pe is not None else 25.0
        pb_val = pb if pb is not None else 3.0
        
        # Normalize: lower PE/PB -> higher score
        # We'll use a threshold-based linear score: 5 is great, 30 is bad.
        s_pe = _linear_score(1/pe_val if pe_val > 0 else 30, 1/30, 1/5) if pe_val > 0 else 0
        # To keep it simple and robust:
        s_pe = max(0, min(100, (30 - pe_val) / (30 - 5) * 100)) if 5 <= pe_val <= 30 else (100 if pe_val < 5 else 0)
        s_pb = max(0, min(100, (3 - pb_val) / (3 - 0.5) * 100)) if 0.5 <= pb_val <= 3 else (100 if pb_val < 0.5 else 0)
        
        return (s_pe + s_pb) / 2

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score based on ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # Higher ROE is better
        roe_score = max(0, min(100, (roe + 0.2) / 0.4 * 100)) if roe < 0.4 else 100
        # Lower debt is better (assuming debt_to_equity)
        debt_score = max(0, min(100, (1.0 - debt) / 1.0 * 100)) if debt < 1.0 else 0
        
        return (roe_score * 0.7) + (debt_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Computes growth score based on ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        
        # PEG: lower is better (growth relative to valuation)
        peg_score = max(0, min(100, (2.0 - peg) / 2.0 * 100)) if peg < 2.0 else 0
        # ROE as a proxy for growth potential/efficiency
        roe_score = max(0, min(100, (roe + 0.2) / 0.4 * 100)) if roe < 0.4 else 100
        
        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Ranks all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results