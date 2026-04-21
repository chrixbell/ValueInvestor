"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more influence from quality/growth
    "quality": 0.25,   # Increased to capture stability
    "growth": 0.25,    # Increased to capture upside potential
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

    def score(self, result: Screening_result) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        # Placeholder for internal logic if needed; actual scoring happens via sub-methods
        pass

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher dividend yield → higher score."""
        # Use a mix of PE and PB for valuation. Avoid division by zero/None.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        div = result.valuation.dividend_yield

        # Penalize negative PE (loss making) by assigning a low base
        pe_val = pe if (pe is not None and pe > 0) else 999.0
        pb_val = pb if (pb is not None and pb > 0) else 999.0
        div_val = div if (div is not None and div > 0) else 0.0

        # Score components
        # We target a range where PE of 5-15 and PB of 1-3 are good.
        s_pe = _linear_score(pe_val, 5.0, 40.0)
        s_pb = _linear_score(pb_val, 1.0, 10.0)
        s_div = _linear_score(div_val, 5.0, 10.0)

        # If PE/PB are extreme (999), they will naturally result in 0 score via clamping
        return (s_pe * 0.4) + (s_pb * 0.3) + (s_div * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe or 0.0
        margin = result.financials.net_margin or 0.0
        debt_eq = result.financials.debt_to_equity or 0.0
        current_ratio = result.financials.current_ratio or 1.0

        # ROE and Margin (Higher is better)
        s_roe = _linear_score(roe, 15.0, -5.0)
        s_margin = _linear_score(margin, 15.0, -5.0)
        
        # Leverage (Lower is better)
        s_debt = _linear_score(debt_eq, 0.5, 3.0)
        s_current = _linear_score(current_ratio, 2.0, 0.5)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.2) + (s_current * 0.1)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG → higher score."""
        roe = result.financials.roe or 0.0
        peg = result.valuation.peg_ratio

        # PEG is tricky with negatives; handle carefully
        if peg is not None and peg > 0:
            s_peg = _linear_score(peg, 0.5, 3.0)
        else:
            s_peg = 50.0 # Neutral for invalid/negative PEG

        # ROE as a proxy for growth capacity
        s_roe = _linear_score(roe, 20.0, -5.0)

        return (s_peg * 0.5) + (s_roe * 0.5)

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        # We need to define the sub-score methods explicitly for this class structure
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
            # Using a small epsilon to prevent log(0) issues if we were using logs, 
            # but here we use power-based geometric mean. 
            # We map score to [0, 1] for the product.
            normalized_val = max(0.0, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The formula (product_of_powers) already accounts for weights in the exponent.
            # If we want (S1^w1 * S2^w2)^ (1/sum_w), we use:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by score descending (higher is better)
        ranked_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(ranked_results):
            # Assign rank (1-based)
            res.rank = i + 1
            
        return ranked_results