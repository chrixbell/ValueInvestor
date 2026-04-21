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
    "quality": 0.25,   # Increased to prioritize stable companies
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
            # Use a small epsilon to prevent zero-score annihilation in geometric mean
            # while still allowing low scores to pull down the composite.
            normalized_score = max(0.01, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # For a geometric mean calculation where weights are normalized, 
            # we don't need the (1/total_weight) exponent if weights already sum to 1.
            # However, we keep it for mathematical consistency with non-normalized weights.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield improves score."""
        v_pe = result.valuation.pe_ratio
        v_pb = result.valuation.pb_ratio
        v_div = result.valuation.dividend_yield

        # Handle potential None or negative PE/PB
        pe = v_pe if (v_pe is not None and v_pe > 0) else 20.0
        pb = v_pb if (v_pb is not None and v_pb > 0) else 2.0
        div = v_div if (v_div is not None and v_div > 0) else 0.0

        # Scoring components (lower is better for PE/PB, higher for dividend)
        s_pe = _linear_score(pe, 40.0, 5.0)
        s_pb = _linear_score(pb, 5.0, 1.0)
        s_div = _linear_score(div, 5.0, 0.0)

        return (s_pe * 0.4 + s_pb * 0.4 + s_div * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt improves score."""
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        debt = result.financials.debt_to_equity if (result.financials.debt_to_equity is not None) else 1.0
        margin = result.financials.net_margin if (result.financials.net_margin is not None) else 0.0

        # Scale ROE and Margin (assuming they are decimals, e.g., 0.15 for 15%)
        # We use a multiplier to bring them into a comparable range [0, 100]
        s_roe = _linear_score(roe * 100, 25.0, 0.0)
        s_debt = _linear_score(debt * 100, 100.0, 0.0)
        s_margin = _linear_score(margin * 100, 20.0, 0.0)

        return (s_roe * 0.5 + s_margin * 0.3 + s_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG improves score."""
        roe = result.financials.roe if (result.financials.roe is not None) else 0.0
        peg = result.valuation.peg_ratio if (result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0) else 2.0

        # Growth score focuses on the relationship between returns and valuation
        s_roe = _linear_score(roe * 100, 25.0, 0.0)
        s_peg = _linear_score(peg * 10, 5.0, 0.5) # PEG scaled

        return (s_roe * 0.6 + s_peg * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results