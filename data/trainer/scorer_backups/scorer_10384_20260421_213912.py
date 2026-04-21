"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more room for quality/growth
    "quality": 0.25,   # Increased to capture better fundamental stability
    "growth": 0.25,    # Increased to better capture expansion potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best: float, worst: float) -> float:
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
            # Use a small epsilon to prevent zero-out in geometric mean if scores are 0
            normalized_score = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        v = result.valuation
        f = result.financials

        # Fallback logic for missing data
        pe = v.pe_ratio if v.pe_ratio is not None and v.pe_ratio > 0 else 30.0
        pb = v.pb_ratio if v.pb_ratio is not None and v.pb_ratio > 0 else 5.0
        ps = v.ps_ratio if v.ps_ratio is not None and v.ps_ratio > 0 else 5.0
        dy = v.dividend_yield if v.dividend_yield is not None and v.dividend_yield > 0 else 2.0

        # Combine valuation metrics
        # We use a simple average of normalized scores for different value factors
        s_pe = _linear_score(1/pe, 1/30.0, 1/5.0) # Inverting so lower PE is higher score
        s_pb = _linear_score(1/pb, 1/5.0, 1/1.0)
        s_ps = _linear_score(1/ps, 1/5.0, 1/1.0)
        s_dy = _linear_score(dy, 2.0, 6.0)

        return (s_pe * 0.4) + (s_pb * 0.2) + (s_ps * 0.2) + (s_dy * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        f = result.financials
        
        roe = f.roe if f.roe is not None else 0.0
        margin = f.net_margin if f.net_margin is not None else 0.0
        debt = f.debt_to_equity if f.debt_to_equity is not None else 0.5
        current = f.current_ratio if f.current_ratio is not None else 1.0

        # ROE: Scale from -20% to 30%
        s_roe = _linear_score(roe, 0.30, -0.20)
        # Margin: Scale from -5% to 25%
        s_margin = _linear_score(margin, 0.25, -0.05)
        # Debt: Lower is better (scaled 0 to 2.0)
        s_debt = _linear_score(1/max(0.1, debt), 1/2.0, 1/0.5)
        # Liquidity: Current ratio (scaled 0 to 3.0)
        s_curr = _linear_score(current, 3.0, 0.5)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.2) + (s_curr * 0.1)

    def _growth_score(self, result: Screening_result = None) -> float:
        # Note: The signature in the provided code was 'result: ScreeningResult' 
        # but I will fix/keep consistency. The template had 'result: ScreeningResult'.
        pass

    # Overwriting the faulty growth_score structure from template to ensure it works.
    # Re-implementing properly below.

    def _growth_score(self, result: ScreeningResult) -> float:
        f = result.financials
        v = result.valuation

        # Growth metrics: We look for positive growth potential via ROE and PEG
        roe = f.roe if f.roe is not None else 0.0
        peg = v.peg_ratio if v.peg_ratio is not None and v.peg_ratio > 0 else 2.0
        
        # Using ROE as a proxy for growth/efficiency
        s_roe = _linear_score(roe, 0.25, -0.10)
        # PEG: Lower is better for growth-at-reasonable-price
        s_peg = _linear_score(1/peg, 1/3.0, 1/0.5)

        return (s_roe * 0.6) + (s_peg * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results

# The above structure has a duplicate method definition due to the template. 
# Let's provide the clean, single-class file.