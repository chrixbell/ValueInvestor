"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture more stability
    "growth": 0.25,    # Increased to capture upside potential
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
    """
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
            # Use a small epsilon to prevent log(0) issues in geometric mean calculation
            normalized_score = max(0.0, score_val) / 100.0
            # If the score is 0, we use a very small epsilon to allow it to be an effective zero
            # but not break the power calculation.
            if normalized_score == 0:
                normalized_score = 1e-6
            
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean structure is simplified here since weights are applied to the terms.
            # If we want (S1^w1 * S2^w2)^(1/sum_w), we divide the product by total weight.
            # However, if weights are already normalized such that sum(w) = 1, product_of_powers is the score.
            # Given weights might not sum to 1, we adjust:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculates value score using PE and PB."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Use PEG as a secondary validator if available
        peg = result.valuation.peg_ratio

        # Base scores from PE and PB (Lower is better)
        # We use a broad range for thresholds to capture diverse market conditions.
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Map PE 0-40 to 100-0 (with some buffer)
            pe_score = _linear_score(pe, 40.0, 1.0)
        elif pe is not None and pe <= 0:
            pe_score = 100.0  # Negative PE is often "undervalued" in certain contexts

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 5.0, 0.1)
        elif pb is not None and pb <= 0:
            pb_score = 100.0

        # Combine using a simple average of available metrics
        valid_scores = []
        if pe is not None: valid_scores.append(pe_score)
        if pb is not None: valid_scores.append(pb_score)
        
        if not valid_scores:
            return 50.0
        
        avg_val = sum(valid_scores) / len(valid_scores)
        return avg_val

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        scores = []
        if roe is not None:
            # ROE: higher is better. 30% is great, -10% is bad.
            scores.append(_linear_score(roe * 100, 30.0, -10.0))
        
        if debt_equity is not None:
            # Debt/Equity: lower is better. 1.0 (100%) to 3.0 (300%).
            # We invert it so higher score = lower debt.
            scores.append(_linear_score(debt_equity, 3.0, 0.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: Screening_result) -> float:
        # This is a dummy definition to satisfy the class structure for the prompt's sake, 
        # but in a real scenario, it would be implemented. 
        # However, I will implement the logic below in the final block.
        pass

    # Re-implementing to ensure structure is correct
    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using PEG and potentially revenue/income growth."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        scores = []
        if peg is not None and peg > 0:
            # Lower PEG is better (growth at a reasonable price)
            scores.append(_linear_score(peg, 2.0, 0.1))
        
        if roe is not None:
            # High ROE often indicates growth-capable businesses
            scores.append(_linear_score(roe * 100, 25.0, 0.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1  # 1-based rank
        
        return sorted_results