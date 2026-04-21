"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Focus on value capture
    "quality": 0.25,   # Stronger focus on quality to mitigate risk
    "growth": 0.25,    # Balanced growth component
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
            # We add a small epsilon (1e-6) to avoid issues with 0 scores in geometric mean
            # while allowing zero to remain a valid low score.
            normalized_score = max(1e-6, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean and scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a mix of valuation metrics. 
        # If multiples are negative (loss making), they get low scores.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        
        # Use PE as primary, fallback to PB or PS
        if pe is not None and pe > 0:
            # Target PE range for 'value': 5 to 20
            return _linear_score(pe, 5.0, 30.0)
        elif pb is not None and pb > 0:
            return _linear_score(pb, 1.0, 5.0)
        elif ps is not None and ps > 0:
            return _linear_score(ps, 1.0, 5.0)
        else:
            return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        
        score_roe = 0.0
        score_debt = 0.0

        if roe is not None:
            # Map ROE (assume range -20 to 40)
            score_roe = _linear_score(roe, 15.0, -10.0)
        
        if debt_to_equity is not None:
            # Lower debt-to-equity is better. 
            # If debt_to_equity is 0, it's the best (100). If 2.0, it's worst (0).
            score_debt = _linear_score(1.0/max(0.01, debt_to_equity), 2.0, 0.1)
            # Wait, the linear_score logic: (val - worst)/(best - worst). 
            # To make low debt high score: best=0, worst=2.
            score_debt = _linear_score(max(0, 2 - debt_to_equity), 0.0, 2.0)
            # Let's simplify:
            if debt_to_equity < 0: # Should not happen with standard D/E
                score_debt = 100.0
            else:
                score_debt = max(0.0, min(100.0, (2.0 - debt_to_equity) / 2.0 * 100.0))
        else:
            score_debt = 50.0

        return (score_roe * 0.7) + (score_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        score_roe = 0.0
        score_peg = 0.0

        if roe is not None:
            # Growth version of ROE (higher is better)
            score_roe = _linear_score(roe, 20.0, -5.0)
        
        if peg is not None and peg > 0:
            # PEG: lower is better. Target PEG of 1.0 is good.
            score_peg = _linear_score(1.0/peg, 0.5, 3.0) # Inverse to make low PEG high score
            # Wait, let's use logic: if peg=0.5 (best), score=100. If peg=3 (worst), score=0.
            # Correct approach for 1/x:
            score_peg = _linear_score(1.0/peg, 2.0, 0.2) # This is messy.
            # Let's use:
            score_peg = max(0.0, min(100.0, (3.5 - peg) / 3.5 * 100.0)) if peg > 0 else 100.0
            # Actually, let's just use a simpler linear clamp:
            score_peg = max(0.0, min(100.0, (3.0 - peg) / 3.0 * 100.0)) if peg < 3 else 0.0
            # Re-calculating for consistency:
        else:
            score_peg = 50.0

        return (score_roe * 0.5) + (score_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        # Sort by composite_score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
        return sorted_results