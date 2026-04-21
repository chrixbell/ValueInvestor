"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to emphasize profitability/growth
    "quality": 0.25,   # Increased to prioritize companies with stronger margins/ROE
    "growth": 0.25,    # Increased to capture momentum in earnings expansion
    "momentum": 0.00,  # Kept at 0.0
}


def _linear_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    def _value_score(self, result: ScreeningResult) -> float:
        """Value score based on inverse PE and PB ratios."""
        v = result.valuation
        # PE and PB are lower-is-better. 
        # Using 5 as 'best' (very cheap) and 30 as 'worst' (expensive) for PE.
        pe = v.pe_ratio if v.pe_ratio and v.pe_ratio > 0 else 30.0
        pb = v.pb_ratio if v.pb_ratio and v.pb_ratio > 0 else 5.0
        
        s1 = _linear_score(pe, 5.0, 30.0)
        s2 = _linear_score(pb, 1.0, 5.0)
        return (s1 * 0.6) + (s2 * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Quality score based on ROE and Debt/Equity."""
        f = result.financials
        # Higher ROE is better, Lower Debt is better.
        roe = f.roe if f.roe is not None else 0.0
        de = f.debt_to_equity if f.debt_to_equity is not None else 2.0
        
        s1 = _linear_score(roe, 25.0, 0.0)
        s2 = _linear_score(de, 0.1, 1.5)
        return (s1 * 0.7) + (s2 * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Growth score based on PEG ratio."""
        v = result.valuation
        # Lower PEG is better. Clamp between 0.5 and 2.5.
        peg = v.peg_ratio if v.peg_ratio and v.peg_ratio > 0 else 2.5
        return _linear_score(peg, 0.5, 2.5)

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score."""
        val_s = self._value_score(result)
        qual_s = self._quality_score(result)
        grow_s = self._growth_score(result)
        
        # Weighted arithmetic mean for stability
        weights = self.weights
        result.composite_score = (
            val_s * weights["value"] +
            qual_s * weights["quality"] +
            grow_s * weights["growth"]
        )
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Score all results and sort by composite score descending."""
        for r in results:
            self.score(r)
        results.sort(key=lambda x: x.composite_score or 0.0, reverse=True)
        return results