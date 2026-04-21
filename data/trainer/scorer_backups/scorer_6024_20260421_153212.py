"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
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
            # Use a small epsilon to avoid issues with log/zero in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # If product is 0, the result should be 0.
            if product_of_powers > 0:
                composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            else:
                composite_score = 0.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use forward_pe if available, otherwise pe_ratio
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle potential None or negative PE (loss making)
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 1.0

        # Scores: Lower PE/PB is better
        # We use a wide range for mapping
        pe_score = _linear_score(pe, 5.0, 40.0)
        pb_score = _linear_score(pb, 0.5, 10.0)
        
        return (pe_score + pb_score) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE: Higher is better (assuming it's expressed as a percentage like 0.15 or 15.0)
        # We normalize based on typical ranges. If ROE is e.g. 0.15, we scale it.
        # Assuming input can be 0.15 or 15.0, we normalize to a comparable range.
        roe_val = roe * 100 if abs(roe) < 2.0 else roe
        
        # Scale ROE: 20% is great, 0% is bad.
        roe_score = _linear_score(roe_val, 25.0, -5.0)
        
        # Debt/Equity: Lower is better. 
        # If debt_equity is 0.5 (50%), it's good.
        de_val = debt_equity * 100 if abs(debt_equity) < 2.0 else debt_equity
        de_score = _linear_score(de_val, 0.5, 3.0)
        
        return (roe_score + de_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG ratio."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        # If PEG is negative or zero, it might indicate high growth/loss. 
        # We treat very low PEG as best (up to a point).
        if peg <= 0:
            peg = 0.1 # Treat as very attractive for the sake of scoring

        # PEG: Lower is better (typically < 1.0 is good)
        growth_score = _linear_score(peg, 0.5, 3.0)
        
        # We also want to factor in basic profitability/growth if available, 
        # but sticking to PEG as requested by the structure.
        return growth_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results