"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for value
    "quality": 0.25,   # Increased quality focus to filter out "value traps"
    "growth": 0.25,    # Balanced growth component
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst:float) -> float:
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
            # Use a small epsilon to avoid zero issues in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The result of product_of_powers is (S1^w1 * S2^w2...). 
            # To get the weighted geometric mean, we want (S1^w1 * S2^w2...)^(1/sum_w)
            # However, since the weights in _DEFAULT_WEIGHTS sum to 1.0, 
            # product_of_powers is already the correct scale if we just multiply by 100.
            # We apply the power to ensure mathematical correctness for any sum of weights.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use Forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Check for valid positive PE/PB
        valid_pe = pe is not None and pe > 0
        valid_pb = pb is not None and pb > 0

        # If both are valid, we weight them. Otherwise use the available one.
        if valid_pe and valid_pb:
            # We want lower PE/PB to be better. 
            # Map PE [0, 50] and PB [0, 15] to scores.
            # This is a placeholder for the logic; actual implementation below.
            pass

        # Let's use a more robust approach: score components separately and average.
        pe_score = 0.0
        if valid_pe:
            # PE range: 1 to 40 (higher is worse)
            pe_score = _linear_score(pe, 1.0, 40.0) if pe < 40 else 0.0
            # Invert so lower PE is higher score: 100 - pe_score (approx)
            pe_score = 100.0 - pe_score if pe_score > 0 else (100.0 if pe < 1.0 else 0.0)
            # Re-normalize for clarity: lower PE is better.
            if pe <= 1.0: pe_score = 100.0
            elif pe > 40.0: pe_score = 0.0
            else: pe_score = (40.0 - pe) / 39.0 * 100.0

        pb_score = 0.0
        if valid_pb:
            if pb <= 0.5: pb_score = 100.0
            elif pb > 15.0: pb_score = 0.0
            else: pb_score = (15.0 - pb) / 14.5 * 100.0

        if valid_pe and valid_pb:
            return (pe_score + pb_score) / 2.0
        return pe_score if valid_pe else pb_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5

        # ROE: Higher is better (Assume 30% is great, -10% is bad)
        roe_score = _linear_score(roe, 30.0, -10.0)
        
        # Debt/Equity: Lower is better (Assume 0 is best, 2.0 is worst)
        de_score = _linear_score(debt_equity, 0.0, 2.0)
        # Flip de_score because _linear_score treats higher as better, but for debt, lower is better.
        # Wait, the _linear_score logic: if best=0 and worst=2, then 0 -> 100, 2 -> 0. Correct.
        
        # Combine: weighted average of ROE and low leverage
        return (roe_score * 0.7 + de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and Revenue Growth (via ROE proxy)."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG: Lower is better (Targeting 0 to 2.0)
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 0.5, 2.5)
            # If peg is very low (e.g. 0.1), score might exceed 100 or be weird
            peg_score = max(0.0, min(100.0, peg_score))
        elif peg is not None and peg <= 0:
            # Negative PEG usually implies negative growth, but in some contexts it's "undervalued growth"
            # For this model, we treat non-positive PEG as a low score to avoid division errors
            peg_score = 10.0 
        else:
            peg_score = 50.0

        # ROE is already used in quality, but we use it here as a proxy for growth potential
        # We scale it differently to emphasize growth-oriented stocks.
        roe_growth_score = _linear_score(roe, 20.0, -5.0)

        return (peg_score * 0.6 + roe_growth_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results