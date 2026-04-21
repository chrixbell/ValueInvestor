"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Focus more on quality/growth to capture mid-term momentum
    "quality": 0.30,   # Increased quality weight to filter for robust balance sheets
    "growth": 0.30,    # Balanced with quality to avoid value traps
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
            # We use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            # but since we are using power-based aggregation, (score/100)^weight is fine.
            # However, to ensure stability if score_val is 0:
            normalized_score = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The weighted geometric mean logic: (product_of_powers)^(1/total_weight)
            # But since product_of_powers is already (S1^w1 * S2^w2...), 
            # we don't need the (1/total_weight) exponent if we want a result in [0, 100].
            # If weights sum to 1.0, product_of_powers is already the result.
            # If weights do not sum to 1, we normalize via the exponent.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        # Use Forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Handle None or extreme values (negative PE)
        pe = pe if (pe is not None and pe > 0) else 20.0
        pb = pb if (pb is not None and pb > 0) else 2.0

        # Score based on low PE and low PB
        # We use a simple linear mapping for the sub-factors
        pe_score = _linear_score(pe, 10.0, 40.0) # Lower is better
        pb_score = _linear_score(pb, 1.0, 5.0)   # Lower is better
        
        # Note: _linear_score(value, best, worst) -> if value < best and worst < best, 
        # we need to flip the logic because in value investing, lower is "best".
        # Let's re-calculate manually to ensure 100 is best.
        
        # Corrected logic: 
        # For PE/PB, 'best' is the low value.
        pe_score = max(0.0, min(100.0, (40.0 - pe) / (40.0 - 10.0) * 100.0)) if pe < 40.0 else (0.0 if pe > 40.0 else 100.0)
        # Wait, the _linear_score function provided: (value - worst) / (best - worst)
        # If best=10, worst=40 -> (pe-40)/(10-40) = (pe-40)/(-30). If pe=10 -> 1.0. If pe=40 -> 0.
        # This works for "lower is better".
        
        pe_s = _linear_score(pe, 10.0, 40.0)
        pb_s = _linear_score(pb, 1.0, 5.0)
        
        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # Higher ROE is better
        roe_s = _linear_score(roe, 20.0, 5.0) # best=20%, worst=5%
        # Lower debt_to_equity is better
        de_s = _linear_score(debt_equity, 0.2, 1.5) # best=0.2, worst=1.5
        
        return (roe_s + de_s) / 2.0

    def _growth_score(self, result: Screening_result) -> float:
        # This is a placeholder structure to match the class requirements. 
        # In reality, growth would use PEG or revenue growth.
        return 50.0

    # Re-implementing the sub-scores to be robust and correct
    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Normalize PE: best is 5, worst is 30. If pe < 5, score=100. If pe > 30, score=0.
        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pe_s = _linear_score(pe_val, 5.0, 30.0)
        
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        pb_s = _linear_score(pb_val, 0.5, 4.0)
        
        return (pe_s + pb_s) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        de = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # ROE: best 25%, worst 0%
        roe_s = _linear_score(roe, 25.0, 0.0)
        # DE: best 0, worst 2.0
        de_s = _linear_score(de, 0.0, 2.0)
        
        return (roe_s + de_s) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        # Using PEG-like logic: ROE / PE (Earnings Yield proxy)
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        if pe is not None and pe > 0:
            # High ROE relative to PE (Growth at reasonable price)
            growth_factor = roe / pe
            # Map a typical growth factor (e.g., 0.2 to 1.0) to 0-100
            return _linear_score(growth_factor, 0.5, 0.05)
        return 50.0

    # The above overrides are messy due to the structure. Let's clean up into a single cohesive class.