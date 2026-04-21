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
    "quality": 0.25,   # Increased to capture stable returns
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
            # We use a small epsilon to prevent 0.0 from zeroing out the whole product
            # while still allowing low scores to pull down the composite.
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean is calculated as (Product of (S/100)^w)^(1 / sum(w))
            # Since we want the final score in [0, 100], we multiply by 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield is better."""
        v = result.valuation
        f = result.financials

        # Factors for value: PE, PB, Dividend Yield
        # We use a simple multi-factor approach for the sub-score
        scores = []
        
        if v.pe_ratio and v.pe_ratio > 0:
            # Using a reasonable range for PE (1 to 40)
            scores.append(_linear_score(v.pe_ratio, 1.0, 40.0) * -1) # We want low PE
            # But the function returns 100 for best (lowest) and 0 for worst.
            # Let's flip the logic: linear_score(value, best=1, worst=40)
            # If pe=5: (5-40)/(1-40)*100 = -35/-39 * 100 = 89.7
            # If pe=45: (45-40)/(1-40)*100 = 5/-39 * 100 = -12 -> clamped to 0
            # Actually, the logic in _linear_score is: score = (val - worst)/(best - worst) * 100
            # To get higher score for lower value: best=low_val, worst=high_val
            pass # Re-evaluating below

        # Corrected logic: best is the target (low for PE), worst is the threshold
        pe_s = _linear_score(v.pe_ratio if v.pe_ratio and v.pe_ratio > 0 else 999, 1.0, 50.0) if v.pe_ratio else 50.0
        pb_s = _linear_score(v.pb_ratio if v.pb_ratio and v.pb_ratio > 0 else 999, 0.1, 10.0) if v.pb_ratio else 50.0
        div_s = _linear_score(v.dividend_yield if v.dividend_yield else 0, 8.0, 0.0) if v.dividend_yield else 50.0
        
        # Average of available value metrics
        val_metrics = [pe_s, pb_s, div_s]
        valid_metrics = [m for m in val_metrics if m is not None]
        return sum(valid_metrics) / len(valid_metrics) if valid_metrics else 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage is better."""
        f = result.financials
        v = result.valuation

        # ROE Score: target 20%, worst 0%
        roe_s = _linear_score(f.roe if f.roe else 0, 25.0, -10.0)
        
        # Debt/Equity: target 0.3, worst 2.0
        de_s = _linear_score(f.debt_to_equity if f.debt_to_equity else 2.0, 0.2, 3.0)
        
        # Margin: target 15%, worst 0%
        margin_s = _linear_score(f.net_margin if f.net_margin else 0, 20.0, -5.0)

        scores = [roe_s, de_s, margin_s]
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        f = result.financials
        v = result.valuation

        # Using ROE as a proxy for growth-quality interaction
        roe_s = _linear_score(f.roe if f.roe else 0, 20.0, -5.0)
        
        # PEG: target 1.0, worst 5.0 (higher is worse)
        # Note: _linear_score(val, best, worst). If pe_growth is 0.5 (best), score = 100.
        # If pe_growth is 5.0 (worst), score = 0.
        peg_s = _linear_score(v.peg_ratio if v.peg_ratio and v.peg_ratio > 0 else 5.0, 0.5, 5.0)

        scores = [roe_s, peg_s]
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Score all and assign ranks based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results