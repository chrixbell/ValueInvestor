"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced weight to allow quality/growth more influence
    "quality": 0.25,   # Increased to reward stable earnings/low debt
    "growth": 0.30,    # Increased to reward expansion potential
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
            # Use a small epsilon to prevent zero-out in geometric mean if score_val is 0
            # but allow it to be very low.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers (S1^w1 * S2^w2...) is already the weighted geometric mean 
            # if we assume weights sum to 1. If not, we adjust via the exponent.
            # Since we want result in [0, 100], and product_of_powers is (S/100)^sum(w)
            # We can just multiply by 100 at the end.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculates value score using PE and PB ratios."""
        # Use forward PE if available, else trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Standardizing ranges for linear scoring
        # PE: 0-30 is a reasonable range for value stocks. Lower is better.
        # PB: 0-5 is a reasonable range for value stocks. Lower is better.
        # We use 1/PE and 1/PB logic implicitly by setting 'best' to low values.
        
        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        
        # Score for PE (lower is better)
        pe_score = _linear_score(1.0/pe_val if pe_val != 0 else 30, 1.0/1.0, 1.0/30.0)
        # Score for PB (lower is better)
        pb_score = _linear_score(1.0/pb_val if pb_val != 0 else 5, 1.0/1.0, 1.0/5.0)
        
        # For value, we also consider dividend yield as a floor/safety.
        div_yield = result.valuation.dividend_yield if result.valuation.dividend_yield is not None else 0.0
        div_score = _linear_score(div_yield, 5.0, 0.0) # 5% is best, 0% is worst

        return (pe_score * 0.4 + pb_score * 0.4 + div_score * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculates quality score using ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE: Higher is better (Targeting 20% as best, 0% as worst)
        roe_score = _linear_score(roe, 20.0, -10.0)
        # Debt/Equity: Lower is better (Targeting 0.2 as best, 2.0 as worst)
        de_score = _linear_score(1.0/max(0.01, debt_equity), 1.0/0.2, 1.0/2.0)
        # Margin: Higher is better (Targeting 15% as best, 0% as worst)
        margin_score = _linear_score(margin, 15.0, -5.0)

        return (roe_score * 0.4 + de_score * 0.3 + margin_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculates growth score using PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG: Lower is better (Targeting 0.5 as best, 3.0 as worst)
        # Handle non-positive PEG by clamping
        peg_val = peg if peg > 0 else 3.0
        peg_score = _linear_score(1.0/peg_val, 1.0/0.5, 1.0/3.0)
        
        # ROE as a proxy for growth efficiency (Higher is better)
        roe_score = _linear_score(roe, 25.0, 0.0)

        return (peg_score * 0.6 + roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results