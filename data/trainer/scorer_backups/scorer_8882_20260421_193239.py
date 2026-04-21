"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow quality/growth influence
    "quality": 0.25,   # Increased weight for quality to improve stability
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            normalized_score = max(1e-6, score_val / 100.0)
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
        # Use forward_pe if available, else pe_ratio
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle potential negative PE (common in loss-making companies)
        # We use a simple clamping: if PE is negative, it's treated as very high/bad for value
        pe_val = pe if pe is not None and pe > 0 else 999.0
        pb_val = pb if pb is not None and pb > 0 else 999.0

        # Weighted average of PE score and PB score
        # For value, lower is better. 
        # We define 'best' as a low PE (e.g., 5) and 'worst' as a high PE (e.g., 40)
        pe_score = _linear_score(pe_val, best=5.0, worst=40.0) if pe_val != 999.0 else 0.0
        pb_score = _linear_score(pb_val, best=1.0, worst=10.0) if pb_val != 999.0 else 0.0
        
        # If PE is invalid, rely on PB and vice versa
        if pe_val == 999.0: return pb_score
        if pb_val == 999.0: return pe_score
        
        return (pe_score * 0.6) + (pb_score * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # ROE: higher is better. Assume 25% is best, 0% is worst for a healthy company.
        roe_score = _linear_score(roe * 100, best=25.0, worst=0.0)
        
        # Debt-to-Equity: lower is better. Assume 0.2 is best, 2.0 is worst.
        # We invert the logic: higher debt -> lower score.
        de_score = _linear_score(debt_equity, best=0.2, worst=2.0) if debt_equity is not None else 50.0
        # Note: _linear_score(value, best=0.2, worst=2.0) where value=0.1 -> (0.1-2)/(0.2-2) = -1.9/-1.8 > 1 -> clamped to 100
        # If value=3.0 -> (3-2)/(0.2-2) = 1/-1.8 -> negative -> clamped to 0
        
        # If debt_equity is None, we rely heavily on ROE. 
        # However, if it's not None, we combine them.
        if debt_equity is None:
            return roe_score
        
        # For de_score, if debt is very high (e.g. 3.0), score should be low.
        # _linear_score(3.0, 0.2, 2.0) -> (3-2)/(0.2-2) = -0.55 -> 0.0
        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using ROE and PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None

        # Growth component: ROE (high is good)
        roe_score = _linear_score(roe * 100, best=20.0, worst=-5.0)

        if peg is None or peg <= 0:
            # If no PEG, we use a fallback or just ROE. 
            # But if PE is negative, peg might be weird. Let's assume PEG needs positive values.
            return roe_score

        # PEG: lower is better (growth relative to valuation). 
        # A PEG of 1.0 is neutral, 0.5 is great, 2.0 is expensive.
        peg_score = _linear_score(peg, best=0.5, worst=2.5)
        
        return (roe_score * 0.4) + (peg_score * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results