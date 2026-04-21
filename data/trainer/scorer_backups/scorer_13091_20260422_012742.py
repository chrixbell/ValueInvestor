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
    "quality": 0.25,   # Increased to prioritize robust balance sheets and profitability
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
            # Use a small epsilon to prevent 0.0 issues in geometric mean when score is exactly 0
            # but keep the weight-based penalty intact.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores [0, 1] is product_of_powers^(1/total_weight)
            # since we are using weights directly in the power.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a combination of PE and PB to capture different valuation aspects.
        # If data is missing, we fall back to single metrics.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Defaulting to reasonable bounds for clamping
        # We use a logic: if PE is provided, use it; else PB; else PS.
        # If all are None, return 50.
        if pe is not None and pe > 0:
            # For PE, lower is better. Range 0-40 covers many profitable companies.
            return _linear_score(pe, 2.0, 40.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 0.5, 10.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, 0.5, 10.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        
        # Quality is often highly correlated with ROE.
        roe_val = roe if roe is not None else 0.0
        # Normalize ROE to a comparable scale (assuming 30% is high, -10% is low)
        roe_score = _linear_score(roe_val * 100, -10.0, 30.0)
        
        # Debt/Equity: lower is better. 
        de_val = debt_equity if debt_equity is not None else 1.0
        # Scale: 2.0 (high debt) to 0.0 (no debt).
        de_score = _linear_score(de_val, 0.0, 2.0)
        # Invert de_score because lower debt = higher score
        de_score = 100.0 - de_score

        return (roe_score * 0.7) + (de_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe
        
        # Use PEG if available as it combines growth and value.
        if peg is not None and peg > 0:
            # PEG of 1.0 is "fair", 0.5 is great, 3.0 is expensive.
            return _linear_score(peg, 0.5, 3.0)
        elif roe is not None:
            # Fallback to ROE if PEG is missing.
            return _linear_score(roe * 100, -5.0, 25.0)
        return 50.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results