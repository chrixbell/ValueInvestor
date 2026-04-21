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
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Using a small epsilon to prevent log(0) issues in geometric mean if score_val is 0
            # but keeping the 0 impact for pure zeros.
            normalized_val = max(score_val, 1e-6) / 100.0
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product_of_powers is (S1/100)^w1 * (S2/100)^w2...
            # To get the geometric mean, we don't need to raise to (1/total_weight) 
            # if the weights themselves are scaled such that sum(weights) = 1.
            # However, to stay robust to non-normalized weights:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # We use a selection of metrics to avoid single-point failure.
        # Prioritize PE/PB but consider others if available.
        vals = []
        if result.valuation.pe_ratio and result.valuation.pe_ratio > 0:
            vals.append(result.valuation.pe_ratio)
        if result.valuation.pb_ratio and result.valuation.pb_ratio > 0:
            vals.append(result.valuation.pb_ratio)
        if result.valuation.ps_ratio and result.valuation.ps_ratio > 0:
            vals.append(result.valuation.ps_ratio)
        if result.valuation.ev_to_ebitda and result.valuation.ev_to_ebitda > 0:
            vals.append(result.valuation.ev_to_ebitda)

        if not vals:
            return 50.0

        # For value, lower is better. We use a simple rank-like approach within the function
        # but since we don't have the full list here, we assume reasonable bounds.
        # In a real scenario, these thresholds would be dynamic based on the universe.
        # Using fixed reasonable bounds for a single-stock score calculation context:
        avg_val = sum(vals) / len(vals)
        # Map low PE/PB to high score. 0-5 range -> 100, 40+ -> 0
        return _linear_score(avg_val, best=5.0, worst=40.0)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE component
        roe_s = _linear_score(roe, best=0.20, worst=0.0)
        # Debt component (Lower is better)
        debt_s = _linear_score(debt, best=0.2, worst=1.5)
        # Margin component
        margin_s = _linear_score(margin, best=0.15, worst=0.0)

        return (roe_s * 0.4 + debt_s * 0.3 + margin_s * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher growth potential (PEG/ROE relation)."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.1
        
        # PEG-based score (Lower is better)
        peg_s = _linear_score(peg, best=0.5, worst=3.0)
        # ROE as a growth proxy
        roe_s = _linear_score(roe, best=0.25, worst=0.0)

        return (peg_s * 0.6 + roe_s * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        
        return sorted_results