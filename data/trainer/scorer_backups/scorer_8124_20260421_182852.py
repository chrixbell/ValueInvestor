"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,     # Reduced value weight to allow more room for quality/growth
    "quality": 0.30,   # Increased quality to capture more stable returns
    "growth": 0.30,    # Balanced growth with quality
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

        # Use an additive weighted mean for better stability and to prevent 
        # a single zero-score from wiping out the entire composite score.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield improve the score."""
        # Use forward_pe if available, else pe_ratio
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Handle case where PE is negative or None
        pe_val = pe if (pe is not None and pe > 0) else 20.0
        pb_val = pb if (pb is not None and pb > 0) else 2.0
        div_val = div if (div is not None and div > 0) else 0.0

        # Score components [0, 100]
        # Lower PE is better (clamp at 30)
        score_pe = _linear_score(pe_val, 5.0, 30.0)
        # Lower PB is better (clamp at 10)
        score_pb = _linear_score(pb_val, 1.0, 10.0)
        # Higher dividend is better
        score_div = _linear_score(div_val, 2.0, 8.0)

        return (score_pe * 0.4 + score_pb * 0.3 + score_div * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower leverage improve the score."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE score: better if higher (range -10% to 30%)
        score_roe = _linear_score(roe * 100, -10.0, 30.0)
        # Debt/Equity: better if lower (range 0 to 2)
        score_debt = _linear_score(debt_equity, 0.0, 2.0)
        # Margin: better if higher (range -5% to 25%)
        score_margin = _linear_score(margin * 100, -5.0, 25.0)

        return (score_roe * 0.4 + score_debt * 0.3 + score_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG improve the score."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # Ensure PEG is positive for scoring logic
        peg_val = peg if (peg is not None and peg > 0.1) else 5.0
        roe_val = roe * 100 if roe is not None else 0.0

        # PEG score: lower is better (range 0.5 to 3)
        score_peg = _linear_score(peg_val, 0.5, 3.0)
        # ROE component for growth (range -10% to 40%)
        score_roe = _linear_score(roe_val, -10.0, 40.0)

        return (score_peg * 0.5 + score_roe * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort all results by composite score and assign ranks."""
        for r in results:
            r.score(r)
        
        # Sort descending by score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results