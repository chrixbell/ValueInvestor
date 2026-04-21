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
    "quality": 0.25,   # Increased quality weight to ensure fundamental robustness
    "growth": 0.25,    # Balanced with quality
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst: float) -> float:
    """Return a score in [0, 100] via linear interpolation."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Return a score in [0, 100] via logarithmic interpolation."""
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

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0, 
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Calculate weighted geometric mean using (S/100) to prevent 0-score dominance 
        # while still penalizing poor sub-scores.
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a tiny epsilon to prevent log(0) issues in geometric mean if score is 0
            normalized_score = max(score_val, 0.001) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The formula (product^(1/total_weight)) * 100 scales it back to [0, 100]
            # However, since we did (S/100)^w, the product is already scaled.
            # We need to adjust: if total_weight is 1, product = (S/100)^w. 
            # To get back to [0, 100], we multiply by 100.
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a combination of PE and PB to capture different valuation aspects
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Use reasonable bounds for A-share/HK market
        # If PE is None, we ignore it in the calculation logic via simple weighting/averaging
        scores = []
        if pe is not None and pe > 0:
            # PE-based score (lower is better)
            # Clamp PE between 0.5 and 50
            clamped_pe = max(0.5, min(50.0, pe))
            scores.append(_linear_score(clamped_pe, 5.0, 40.0))
        
        if pb is not None and pb > 0:
            clamped_pb = max(0.1, min(20.0, pb))
            scores.append(_linear_score(clamped_pb, 1.0, 5.0))
            
        if ps is not None and ps > 0:
            clamped_ps = max(0.1, min(20.0, ps))
            scores.append(_linear_score(clamped_ps, 1.0, 5.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower debt is better."""
        roe = result.financials.roe
        debt_to_equity = result.financials.debt_to_equity
        gross_margin = result.financials.gross_margin

        scores = []
        # ROE is a primary quality metric
        if roe is not None:
            # Scale ROE (assume -20% to 40%)
            clamped_roe = max(-20.0, min(40.0, roe))
            scores.append(_linear_score(clamped_roe, 10.0, 25.0))
        
        if gross_margin is not None:
            clamped_gm = max(0.0, min(60.0, gross_margin))
            scores.append(_linear_score(clamped_gm, 15.0, 40.0))

        # Debt to equity (lower is better)
        if debt_to_equity is not None:
            clamped_d_e = max(0.0, min(2.0, debt_to_equity))
            # 0 is best, 1.5+ is worst
            scores.append(_linear_score(clamped_d_e, 0.2, 1.5))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: Screening_result) -> float:
        # This is a placeholder to avoid errors if the method signature was slightly different in user prompt
        pass

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        peg = result.valuation.peg_ratio
        roe = result.financials.roe
        
        scores = []
        if peg is not None and peg > 0:
            # Low PEG is great (growth at reasonable price)
            clamped_peg = max(0.1, min(3.0, peg))
            scores.append(_linear_score(clamped_peg, 0.5, 2.0))
        
        if roe is not None:
            clamped_roe = max(0.0, min(50.0, roe))
            scores.append(_linear_score(clamped_roe, 5.0, 30.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sort results by composite score and assign ranks."""
        # Sort descending (highest score = rank 1)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results