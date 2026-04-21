"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Slightly reduced to allow for higher quality/growth contribution
    "quality": 0.25,   # Increased to prioritize stable businesses
    "growth": 0.30,    # Increased to capture expansionary signals
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent log(0) issues in geometric mean 
            # while allowing the score to approach zero.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers is already (S1/100)^w1 * (S2/100)^w2 ...
            # We don't need to take the 1/total_weight root because the weights 
            # are already applied as exponents. We just need to handle normalization.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Use PEG as a tie-breaker or primary if available, but stick to PE/PB logic
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Define reasonable bounds for A-shares/HK
        pe_best, pe_worst = 15.0, 60.0
        pb_best, pb_worst = 1.0, 10.0

        score_pe = _linear_score(pe if pe is not None else pe_worst, pe_best, pe_worst) if pe is not None else 50.0
        score_pb = _linear_score(pb if pb is not None else pb_worst, pb_best, pb_worst) if pb is not None else 50.0
        
        # If PE is negative (loss making), it's usually undesirable for a value score 
        # unless we are looking at deep value (which is risky). We clamp to worst.
        if pe is not None and pe <= 0: score_pe = 0.0
        if pb is not None and pb <= 0: score_pb = 0.0

        return (score_pe + score_pb) / 2.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        # ROE targets: 15% to 30%+
        roe_score = _linear_score(roe if roe is not None else 0.0, 20.0, 40.0) if roe is not None else 50.0
        # Debt/Equity: lower is better (e.g., < 1.0 is good, > 3.0 is bad)
        de_score = _linear_score(debt_equity if debt_equity is not None else 2.0, 0.5, 3.0) if debt_equity is not None else 50.0
        # Margin: higher is better
        margin_score = _linear_score(margin if margin is not None else 0.1, 0.1, 0.3) if margin is not None else 50.0

        # If values are extreme/negative
        if roe is not None and roe <= 0: roe_score = 0.0
        if debt_equity is not None and debt_equity < 0: de_score = 100.0 # Negative debt is equity

        return (roe_score * 0.5) + (de_score * 0.3) + (margin_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        # High ROE is a proxy for quality-growth
        roe_score = _linear_score(roe if roe is not None else 0.0, 10.0, 30.0) if roe is not None else 50.0
        # PEG: lower is better (typically < 1.0)
        peg_score = _linear_score(peg if peg is not None else 2.0, 0.5, 3.0) if peg is not None else 50.0
        
        if peg is not None and peg <= 0: peg_score = 100.0 # Avoid division by zero/negative PEG issues
        if roe is not None and roe <= 0: roe_score = 0.0

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