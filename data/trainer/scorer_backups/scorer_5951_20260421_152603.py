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
    "quality": 0.25,   # Increased weight to capture fundamental stability
    "growth": 0.25,    # Increased weight to capture expansion potential
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

        # We use a weighted arithmetic mean for the composite score to prevent 
        # one bad-but-not-zero subscore from zeroing out the entire composite score,
        # while still rewarding high performance across all sectors.
        total_weight = 0.0
        weighted_sum = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            weighted_sum += score_val * weight

        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Computes value score based on PE and PB ratios."""
        # Using a combination of PE and PB to capture value.
        # If PE is negative (loss making), we rely more on PB or just give low score.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Thresholds for 'best' and 'worst' in a typical market context
        # Low PE/PB is better.
        pe_score = 0.0
        if pe is not None and pe > 0:
            # Best PE ~ 10, Worst PE ~ 50 (clamped)
            pe_score = _linear_score(pe, 10.0, 50.0)
            # If PE is very low (e.g., 2), it's better than 10
            if pe < 10.0:
                pe_score = min(100.0, pe_score + (10.0 - pe) * 5.0)
        elif pe is not None and pe <= 0:
            # For negative PE, we use PB as a fallback or assign low score
            pe_score = 0.0

        pb_score = 0.0
        if pb is not None and pb > 0:
            pb_score = _linear_score(pb, 1.0, 5.0)
        elif pb is not None and pb <= 0:
            pb_score = 0.0

        # Combine PE and PB (weighting PE slightly more if it's valid)
        if pe is not None and pe > 0:
            return (pe_score * 0.7) + (pb_score * 0.3)
        return pb_score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Computes quality score based on ROE and Debt/Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity

        roe_score = 0.0
        if roe is not None:
            # High ROE is better. Best 25%, Worst 0% (clamped)
            roe_score = _linear_score(roe, 25.0, 0.0)
            # Handle cases where ROE is higher than 25%
            if roe > 25.0:
                roe_score = min(100.0, (roe / 25.0) * 50.0)

        de_score = 0.0
        if debt_equity is not None:
            # Lower Debt/Equity is better. Best 0.2, Worst 2.0
            de_score = _linear_score(debt_equity, 0.2, 2.0)
        elif debt_equity is None:
            de_score = 50.0

        # Quality is a balance of profitability and solvency
        if roe is not None:
            return (roe_score * 0.7) + (de_score * 0.3)
        return de_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Computes growth score based on ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # Growth is often captured by ROE in value-investing contexts
        roe_score = 0.0
        if roe is not None:
            roe_score = _linear_score(roe, 20.0, -5.0)

        peg_score = 0.0
        if peg is not None and peg > 0:
            # Low PEG is good (growth relative to PE)
            peg_score = _linear_score(peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            peg_score = 0.0

        if peg is not None and peg > 0:
            return (peg_score * 0.6) + (roe_score * 0.4)
        return roe_score

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending (higher is better)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results