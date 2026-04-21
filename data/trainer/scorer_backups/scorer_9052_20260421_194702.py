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
    "quality": 0.25,   # Increased to prioritize stable companies
    "growth": 0.25,    # Balanced with quality
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
            # Use a small epsilon to prevent zero-out in geometric mean if score is 0
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # If product is 0, we handle it to avoid math errors.
            if product_of_powers == 0:
                composite_score = 0.0
            else:
                # We use the weighted product directly as an aggregation method.
                # Since weights sum to 1 (or are normalized), the product of (s/100)^w
                # is already effectively the geometric mean logic.
                composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB is better."""
        # Using a combination of PE and PB to capture valuation. 
        # We use PEG as a secondary check if available.
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Fallback values to prevent division by zero or extreme skews
        pe = pe if (pe is not None and pe > 0) else 20.0
        pb = pb if (pb is not None and pb > 0) else 2.0
        
        # Primary score: Low PE and PB are good.
        # We want to map low values to high scores.
        score_pe = _linear_score(1.0, 5.0, 40.0) if pe < 40 else _linear_score(pe, 5.0, 40.0) # Placeholder logic
        # Let's use a more robust approach:
        score_pe = max(0.0, min(100.0, 100.0 - (pe / 0.5 if pe > 0 else 0))) # This is too aggressive
        
        # Re-implementing standard linear mapping for stability:
        # We'll define 'best' as 1 and 'worst' as 50 for PE.
        score_pe = _linear_score(pe, 1.0, 50.0)
        score_pb = _linear_score(pb, 1.0, 10.0)
        
        # If PEG is available and low, it's a strong value signal.
        if peg is not None and 0 < peg < 1:
            return (score_pe * 0.4) + (score_pb * 0.6)
        return (score_pe * 0.5) + (score_pb * 0.5)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        
        # ROE score (assume 30% is great, -10% is bad)
        score_roe = _linear_score(roe * 100, 30.0, -10.0)
        # Margin score (assume 20% is great, 0% is bad)
        score_margin = _linear_score(margin * 100, 20.0, 0.0)
        # Debt score (lower is better)
        score_debt = _linear_score(debt * 100, 20.0, 150.0)
        
        return (score_roe * 0.4) + (score_margin * 0.3) + (score_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        
        # Higher ROE is a proxy for growth potential in this context
        score_roe = _linear_score(roe * 100, 25.0, -5.0)
        # Low PEG is better for growth-at-reasonable-price
        score_peg = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0
        
        return (score_roe * 0.5) + (score_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            r.composite_score = 0.0 # Reset to ensure fresh calculation if needed
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
        return results

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Handle None/Zero
        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        peg_val = peg if (peg is not None and peg > 0) else 2.0

        # Map PE: 1 -> 100, 40 -> 0
        s_pe = _linear_score(pe_val, 1.0, 40.0)
        # Map PB: 0.5 -> 100, 8 -> 0
        s_pb = _linear_score(pb_val, 0.5, 8.0)
        # Map PEG: 0.5 -> 100, 3 -> 0
        s_peg = _linear_score(peg_val, 0.5, 3.0)

        # Weighting: PEG is a strong validator for value
        return (s_pe * 0.35) + (s_pb * 0.35) + (s_peg * 0.30)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        # ROE: 25% is best, -10% is worst
        s_roe = _linear_score(roe * 100, 25.0, -10.0)
        # Margin: 15% is best, 0% is worst
        s_margin = _linear_score(margin * 100, 15.0, 0.0)
        # Debt: 30% is best (low), 150% is worst
        s_debt = _linear_score(debt * 100, 30.0, 150.0)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        # Growth is often tied to ROE and PEG in these datasets
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0
        
        # Higher ROE (growth-oriented)
        s_roe = _linear_score(roe * 100, 30.0, -5.0)
        # Lower PEG (growth at reasonable price)
        s_peg = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0

        return (s_roe * 0.5) + (s_peg * 0.5)

    # Redefining methods to ensure they are clean and handle the logic above
    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        peg_val = peg if (peg is not None and peg > 0) else 2.0

        s_pe = _linear_score(pe_val, 1.0, 40.0)
        s_pb = _linear_score(pb_val, 0.5, 8.0)
        s_peg = _linear_score(peg_val, 0.5, 3.0)

        return (s_pe * 0.4) + (s_pb * 0.3) + (s_peg * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        s_roe = _linear_score(roe * 100, 25.0, -10.0)
        s_margin = _linear_score(margin * 100, 15.0, 0.0)
        s_debt = _linear_score(debt * 100, 30.0, 150.0)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0

        s_roe = _linear_score(roe * 100, 30.0, -5.0)
        s_peg = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0

        return (s_roe * 0.5) + (s_peg * 0.5)

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        peg_val = peg if (peg is not None and peg > 0) else 2.0

        s_pe = _linear_score(pe_val, 1.0, 40.0)
        s_pb = _linear_score(pb_val, 0.5, 8.0)
        s_peg = _linear_score(peg_val, 0.5, 3.0)

        return (s_pe * 0.4) + (s_pb * 0.3) + (s_peg * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        s_roe = _linear_score(roe * 100, 25.0, -10.0)
        s_margin = _linear_score(margin * 100, 15.0, 0.0)
        s_debt = _linear_score(debt * 100, 30.0, 150.0)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0

        s_roe = _linear_score(roe * 100, 30.0, -5.0)
        s_peg = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0

        return (s_roe * 0.5) + (s_peg * 0.5)

    # Re-cleaning the class to remove duplicate method definitions attempt
    # The above was a thinking process. Let's provide the final clean version.

class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    def score(self, result: ScreeningResult) -> ScreeningResult:
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": 50.0,
        }

        weighted_scores = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores[k] = score_val

        if not weighted_scores:
            result.composite_score = 50.0
            return result

        # Calculate weighted geometric mean: Product of (score/100)^weight
        total_weight = sum(self.weights[k] for k in weighted_scores)
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        product_of_powers = 1.0
        for k, score_val in weighted_scores.items():
            # Normalize to [0, 1] for geometric mean calculation
            normalized_val = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_val ** self.weights[k])

        # The result is currently in [0, 1] range scaled by weights.
        # To get it back to [0, 100], we scale by 100.
        # Note: product_of_powers is (s1/100)^w1 * (s2/100)^w2...
        # Since sum(weights) might not be 1.0, we normalize by total_weight
        # but the exponentiation logic is: result = 100 * (product)^(1/sum_w)
        # However, if we want a simple weighted product:
        result.composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        pe_val = pe if (pe is not None and pe > 0) else 30.0
        pb_val = pb if (pb is not None and pb > 0) else 5.0
        peg_val = peg if (peg is not None and peg > 0) else 2.0

        s_pe = _linear_score(pe_val, 1.0, 40.0)
        s_pb = _linear_score(pb_val, 0.5, 8.0)
        s_peg = _linear_score(peg_val, 0.5, 3.0)

        return (s_pe * 0.4) + (s_pb * 0.3) + (s_peg * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        s_roe = _linear_score(roe * 100, 25.0, -10.0)
        s_margin = _linear_score(margin * 100, 15.0, 0.0)
        s_debt = _linear_score(debt * 100, 30.0, 150.0)

        return (s_roe * 0.4) + (s_margin * 0.3) + (s_debt * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 2.0

        s_roe = _linear_score(roe * 100, 30.0, -5.0)
        s_peg = _linear_score(peg, 0.5, 3.0) if peg > 0 else 100.0

        return (s_roe * 0.5) + (s_peg * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results