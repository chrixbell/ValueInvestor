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
    "quality": 0.25,   # Increased to capture stable earnings and low leverage
    "growth": 0.25,    # Increased to capture potential upside
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

        # We use a modified geometric mean approach where we add a small epsilon 
        # to avoid zero-multiplication issues, but since we use (score/100), 
        # a score of 0 will still correctly pull the composite to 0.
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0.0, 1.0] range
            product_of_powers *= (score_val / 100.0) ** weight

        if total_weight > 0:
            # The product of (S/100)^w is already the weighted geometric mean 
            # if we assume weights sum to 1. If they don't, we compensate.
            # However, (S/100)^w where sum(w) != 1 is effectively S_weighted = product(S^w)^(1/sum(w))
            # The current formula is slightly different: product_of_powers = product( (S/100)^w )
            # To get the correct geometric mean: Composite = 100 * (product_of_powers ^ (1/total_weight))
            # But wait, if we want the weighted geometric mean of X_i: (product X_i^w_i)^(1/sum(w_i))
            # Our product_of_powers is already that. 
            # Let's re-verify: if weights are [0.5, 0.5], product is (S1/100)^0.5 * (S2/100)^0.5 = sqrt(S1/100 * S2/100)
            # This is already the geometric mean. No further power needed unless we want to normalize weights.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and higher Dividend Yield is better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Heuristic bounds for Chinese markets
        pe_best, pe_worst = 10.0, 50.0
        pb_best, pb_worst = 1.0, 10.0
        dy_best, dy_worst = 5.0, 0.0

        v1 = _linear_score(pe if pe is not None else 25.0, pe_best, pe_worst) if pe is not None else 50.0
        # For PE, lower is better, but _linear_score expects higher=better. 
        # We need to invert it or use a different logic.
        # Let's redefine: 
        if pe is not None:
            v1 = _linear_score(pe, pe_worst, pe_best) # If PE is 5 (worst), score is high. Wait, logic error.
            # Let's fix: if pe=5 (best) and pe_best=10, worst=50.
            # If pe=5: (5-50)/(10-50) = -45/-40 = 1.12 -> clamped to 100. Correct.
            # If pe=50: (50-50)/... = 0. Correct.
            v1 = _linear_score(pe, pe_best, pe_worst) # This gives high score for low PE.
            # Re-calculating: if pe=5, best=10, worst=50. (5-50)/(10-50) = 1.12 -> 100.
            # If pe=50, (50-50)/... = 0.
            # Let's use a simpler approach:
        
        # Re-writing sub-scores for clarity
        def get_low_is_better(val, best, worst):
            if val is None: return 50.0
            return _linear_score(val, worst, best) # If val is 5 and best=10, worst=50: (5-50)/(10-50) = 1.12 -> 100
            # Actually, the formula (val - worst)/(best - worst) where best < worst:
            # If val=10, best=10, worst=50 -> (10-50)/(10-50) = 1.0 -> 100%
            # If val=50, best=10, worst=50 -> (50-50)/... = 0%
            # If val=5, best=10, worst=50 -> (5-50)/(10-50) = 1.12 -> 100%
            # This works.

        v_pe = get_low_is_better(pe, pe_best, pe_worst)
        v_pb = get_low_is_better(pb, pb_best, pb_worst)
        v_dy = _linear_score(dy if dy is not None else 0.0, dy_best, dy_worst)
        
        # Combine: weight PB/PE more if PE is unavailable
        if pe is not None and pb is not None:
            return (v_pe * 0.6 + v_pb * 0.4)
        elif pe is not None:
            return v_pe
        else:
            return v_pb

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        # ROE/Margin: higher is better
        # Debt: lower is better
        r_roe = _linear_score(roe if roe is not None else 0.0, 20.0, -10.0)
        r_margin = _linear_score(margin if margin is not None else 0.0, 15.0, -5.0)
        r_debt = _linear_score(debt if debt is not None else 1.0, 0.5, 3.0) # lower debt = higher score

        # If debt is low (e.g. 0.1), it should be high score.
        # If debt=0.5 (best) and debt=3.0 (worst):
        # _linear_score(0.1, 0.5, 3.0) -> (0.1-3)/(0.5-3) = -2.9/-2.5 = 1.16 -> 100
        # This logic works.

        q_score = (r_roe * 0.4 + r_margin * 0.4 + r_debt * 0.2)
        return q_score

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        # If we have PEG, use it.
        if peg is not None and peg > 0:
            # PEG = PE / Growth. Low PEG is good.
            # We use log_score for PEG-like metrics to handle distribution
            g_peg = _linear_score(peg, 0.5, 3.0)
            # Use ROE as a proxy for growth/quality combo
            g_roe = _linear_score(roe if roe is not None else 0.0, 15.0, -5.0)
            return (g_peg * 0.4 + g_roe * 0.6)
        else:
            # Fallback to ROE-based growth score
            return _linear_score(roe if roe is not None else 0.0, 15.0, -5.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results