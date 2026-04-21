"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Adjusted to balance with quality/growth
    "quality": 0.30,   # Increased quality to capture more stable returns
    "growth": 0.25,    # Maintained growth component
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

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0
        
        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


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

        # Using an arithmetic mean for the composite score to prevent 
        # a single zero-score factor from nullifying high performance in others,
        # while still allowing the weights to guide the ranking.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher Dividend Yield are better."""
        v = result.valuation
        f = result.financials

        # Primary value drivers
        pe = v.pe_ratio if v.pe_ratio is not None and v.pe_ratio > 0 else None
        pb = v.pb_ratio if v.pb_ratio is not None and v.pb_ratio > 0 else None
        div = v.dividend_yield if v.dividend_yield is not None and v.dividend_yield > 0 else None

        # Scoring logic
        # We use a simple hierarchy: if PE is available, use it; else PB; else Dividend
        # to ensure we always have a value component.
        scores = []
        if pe:
            # PE 5 is great, 30 is expensive.
            scores.append(_linear_score(pe, 5, 30))
        elif pb:
            # PB 1 is great, 4 is expensive.
            scores.append(_linear_score(pb, 1, 4))
        elif div:
            # Dividend yield (higher is better) - inversion needed for linear_score
            scores.append(_linear_score(div, 5, 0)) # Note: logic flip
            # Wait, the _linear_score is (val-worst)/(best-worst). 
            # For PE/PB: best=low, worst=high. 
            # Let's use a more direct approach for clarity:
        
        # Re-implementing logic to be robust
        s = 50.0
        if pe:
            # If PE is 5 (best), score = (5-30)/(5-30)*100 = 100. If PE is 30 (worst), score=0.
            s = _linear_score(pe, 5, 30)
        elif pb:
            s = _linear_score(pb, 1, 4)
        elif div:
            # For dividend, higher is better. We map best=6% to 100, worst=0% to 0.
            s = (div / 6.0) * 100.0 if div < 6.0 else 100.0
            s = max(0.0, min(100.0, s))
        else:
            # Fallback if no data
            s = 50.0
        return s

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt-to-equity are better."""
        f = result.financials
        roe = f.roe if f.roe is not None else 0.0
        d_e = f.debt_to_equity if f.debt_to_equity is not None else 0.5 # assume neutral
        
        # ROE score: 20% is great, 5% is poor
        roe_s = _linear_score(roe, 20.0, 5.0)
        # Debt score: 0.3 is great, 1.5 is poor
        debt_s = _linear_score(d_e, 0.3, 1.5)
        
        return (roe_s + debt_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG are better."""
        v = result.valuation
        f = result.financials
        peg = v.peg_ratio if v.peg_ratio is not None and v.peg_ratio > 0 else None
        roe = f.roe if f.roe is not None else 0.0

        if peg:
            # PEG 0.5 is great, 2.0 is poor
            g_s = _linear_score(peg, 0.5, 2.0)
        else:
            # Fallback to ROE-based growth proxy if PEG is missing
            g_s = _linear_score(roe, 15.0, 5.0)
            
        return g_s

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort all results by composite score and assign ranks."""
        # Sort descending (highest score first)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results