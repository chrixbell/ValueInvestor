"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for valuation
    "quality": 0.25,   # Increased quality to ensure fundamental stability
    "growth": 0.25,    # Balanced with quality
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

        # We use an arithmetic mean of normalized scores to prevent a single 0 score 
        # (from one bad factor) from zeroing out the entire composite score, 
        # which can happen in geometric means.
        total_weighted_sum = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Adding a small epsilon to avoid issues with 0 in certain math operations, 
            # though arithmetic mean is generally safe.
            total_weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = total_weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield improves score."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Heuristic thresholds for Chinese markets
        pe_score = 0.0
        if pe is not None and pe > 0:
            pe_score = _linear_score(pe, 15.0, 40.0)
        else:
            pe_score = 50.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1.5, 4.0)
        else:
            pb_score = 50.0

        div_score = 0.0
        if div is not None:
            # Dividend yield: higher is better. 0% to 10% range.
            div_score = _linear_score(div, 5.0, 0.0) # Using linear with inverted bounds
            # Correction: _linear_score(val, best, worst). If 5 is best and 0 is worst:
            div_score = (min(max(div, 0.0), 10.0) / 5.0) * 100.0 # Simple clamp
            # Re-evaluating: if div=5, score=100. If div=0, score=0.
            div_score = max(0.0, min(100.0, (div / 5.0) * 100.0 if div is not None else 50.0))
            # Let's stick to a simpler approach for div:
            if div is not None:
                div_score = max(0.0, min(100.0, (div / 6.0) * 100.0))
            else:
                div_score = 50.0

        # Combine value factors
        if pe is not None and pb is not None:
            return (pe_score + pb_score) / 2.0
        elif pe is not None:
            return pe_score
        else:
            return pb_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt-to-equity improves score."""
        roe = result.financials.roe
        d_e = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE is a primary quality driver
        roe_score = 0.0
        if roe is not None:
            # Scale ROE (assuming -20% to 40% range)
            roe_score = max(0.0, min(100.0, (roe + 20) / 60 * 100))
        else:
            roe_score = 50.0

        # Debt to Equity (lower is better)
        de_score = 0.0
        if d_e is not None:
            # Scale D/E (assuming 0 to 2.0 range)
            de_score = _linear_score(d_e, 0.5, 2.0)
        else:
            de_score = 50.0

        # Margin (higher is better)
        margin_score = 0.0
        if margin is not None:
            margin_score = max(0.0, min(100.0, (margin + 0.1) / 0.5 * 100)) # Cap at 50% margin
        else:
            margin_score = 50.0

        # Weighting quality components
        return (roe_score * 0.5) + (de_score * 0.3) + (margin_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG improves score."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # ROE is used in growth as well, but often represents the quality of growth
        roe_score = 0.0
        if roe is not None:
            roe_score = max(0.0, min(100.0, (roe + 20) / 60 * 100))
        else:
            roe_score = 50.0

        peg_score = 0.0
        if peg is not None and peg > 0:
            # PEG: lower is better (1.0 is neutral, <1 is great)
            peg_score = _linear_score(peg, 0.5, 3.0)
        else:
            peg_score = 50.0

        return (roe_score * 0.4) + (peg_score * 0.6)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results