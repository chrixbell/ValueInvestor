"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture more stable returns
    "growth": 0.30,    # Increased to capture higher upside potential
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
            # Use a small epsilon to prevent log(0) issues in geometric mean if score is 0
            normalized_score = max(score_val / 100.0, 1e-6)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean formula is (product of x_i^w_i) ^ (1 / sum(w_i))
            # However, since we are normalizing by 100 at the end, and our weights
            # might not sum to 1.0 in all scenarios (though they do here), 
            # this structure handles the scaling correctly.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use forward PE if available, else trailing PE. 
        # If both are None, fallback to PB.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # We use a ranking-style approach within the sub-score to handle outliers
        # but since we don't have the full list here, we use reasonable thresholds.
        # We prioritize PE then PB.
        if pe is not None and pe > 0:
            # A PE of 5 is great, 30 is expensive.
            return _linear_score(pe, 30.0, 5.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 5.0, 1.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, 3.0, 0.5)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # Quality score component: ROE (higher better) and Debt/Equity (lower better)
        # Convert ROE to a scale: 20% is good, 5% is bad.
        # Convert Debt/Equity to a scale: 1.0 is neutral, 3.0 is bad, 0.2 is great.
        
        # We use a combined approach: ROE score + Debt score
        roe_score = _linear_score(roe * 100, 20.0, 5.0) if roe is not None else 50.0
        
        if debt_equity is not None:
            # Clamp debt to realistic ranges for scoring
            debt_score = _linear_score(debt_equity, 2.0, 0.1)
        else:
            debt_score = 50.0

        return (roe_score + debt_score) / 2.0

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe if result.financials.roe is not None else 0.0

        # PEG is a classic growth-at-reasonable-price metric
        if peg is not None and peg > 0:
            peg_score = _linear_score(peg, 1.5, 0.5)
        else:
            # If PEG is missing, we rely on ROE as a proxy for growth potential in this context
            peg_score = 50.0

        # ROE is already used in quality, but here we treat it as a growth driver
        # (Higher ROE often correlates with high-growth companies)
        roe_score = _linear_score(roe * 100, 25.0, 5.0) if roe is not None else 50.0

        return (peg_score + roe_score) / 2.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results