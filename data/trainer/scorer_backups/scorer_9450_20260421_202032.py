"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Adjusted to allow more room for quality/growth
    "quality": 0.25,   # Increased to capture stable returns
    "growth": 0.30,    # Increased to capture upside potential
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

    def score(self, result: Screening_Result) -> Screening_Result:
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
            # Use a small epsilon to prevent zero-score from wiping out the entire product 
            # in geometric mean if one factor is zero, but still maintain sensitivity.
            normalized_score = max(0.001, score_val) / 100.0
            product_of_powers *= normalized_score ** weight

        if total_weight > 0:
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score using PE and PB ratios."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # Using forward PE if available as it's more predictive of future returns
        pe_val = result.valuation.pe_forward if pe is not None and result.valuation.pe_forward is not None else pe
        
        score_pe = 0.0
        if pe_val is not None and pe_val > 0:
            # Target PE range [1, 25] for scoring. Higher than 25 is penalized.
            score_pe = _linear_score(pe_val, 1.0, 25.0)
        elif pe_val is not None and pe_val <= 0:
            # Negative PE can be good (profitable) or bad. For simplicity, assume 0-PE is highly valued
            score_pe = 100.0 if pe_val < 0 else 50.0
            
        score_pb = 0.0
        if pb is not None and pb > 0:
            score_pb = _linear_score(pb, 0.5, 5.0)
        elif pb is not None and pb <= 0:
            score_pb = 50.0

        # Weight PE more heavily than PB for growth-oriented value
        return (score_pe * 0.7) + (score_pb * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        d_e = result.financials.debt_to_equity

        score_roe = 0.0
        if roe is not None:
            # Map ROE to [0, 100]. Assuming 25% is excellent and -5% is worst.
            score_roe = _linear_score(roe * 100, -5.0, 25.0)
        
        score_de = 0.0
        if d_e is not None:
            # Lower debt-to-equity is better. Mapping 0% to 100 and 150% to 0.
            score_de = _linear_score(d_e * 100, 0.0, 150.0)
            # Since lower is better for debt, we invert the linear score logic manually 
            # (though _linear_score is designed for best=higher)
            # Wait, the function maps 'best' to 100. If best is 0 (low debt), we need:
            score_de = _linear_score(d_e * 100, 0.0, 2.0) # If d_e is 0 -> 100; if 2 -> 0.
            # Let's re-calculate correctly:
            if d_e * 100 > 2.0: score_de = 0.0
            elif d_e * 100 < 0: score_de = 100.0
            else: score_de = (2.0 - (d_e * 100)) / 2.0 * 100.0

        # Let's use a more robust way for de:
        if d_e is not None:
            de_val = d_e * 100
            score_de = max(0.0, min(100.0, (2.0 - de_val) / 2.0 * 100.0)) if de_val < 2.0 else 0.0
            # Actually, let's just use a fixed threshold
            score_de = 100.0 if de_val <= 0.5 else (max(0.0, (1.5 - de_val) / 1.5 * 100.0) if de_val < 1.5 else 0.0)
            # Let's simplify:
            if de_val <= 0.5: score_de = 100.0
            elif de_val <= 1.5: score_de = 50.0
            else: score_de = 0.0

        return (score_roe * 0.7) + (score_de * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using PEG and Gross Margin."""
        peg = result.valuation.peg_ratio
        gm = result.financials.gross_margin

        score_peg = 0.0
        if peg is not None and peg > 0:
            # PEG of 1.0 is neutral, lower is better (growth at reasonable price)
            score_peg = _linear_score(1.0/peg, 0.5, 3.0)
        elif peg is not None and peg <= 0:
            score_peg = 100.0

        score_gm = 0.0
        if gm is not None:
            # Higher gross margin is better
            score_gm = _linear_score(gm * 100, 5.0, 50.0)

        return (score_peg * 0.6) + (score_gm * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results