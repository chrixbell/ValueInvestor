"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced slightly to allow more weight in quality/growth
    "quality": 0.25,   # Increased to reward stable earnings and low leverage
    "growth": 0.25,    # Increased to capture expansionary potential
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

        # Use a small epsilon to prevent zero-product issues in geometric mean 
        # while still allowing low scores to propagate.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] for geometric mean. 
            # Use epsilon to handle zero scores gracefully in a product-based approach.
            normalized_val = max(epsilon, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Calculate the weighted geometric mean scaled back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use PEG as a secondary check if available
        peg = result.valuation.peg_ratio
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio

        # Base score logic: lower PE/PB is better
        # We use a soft-clamped linear approach to avoid extreme outliers
        v_pe = 0.0
        if pe is not None and pe > 0:
            # Mapping PE 0-50 to 100-0
            v_pe = _linear_score(pe, 5.0, 40.0)
        else:
            v_pe = 20.0 # Neutral/low for missing data

        v_pb = 0.0
        if pb is not None and pb > 0:
            v_pb = _linear_score(pb, 1.0, 5.0)
        else:
            v_pb = 20.0

        # If PEG is available and low, it's a strong value signal
        if peg is not None and peg > 0:
            v_peg = _linear_score(peg, 0.5, 2.0)
            return (v_pe * 0.4 + v_pb * 0.4 + v_peg * 0.2)

        return (v_pe * 0.6 + v_pb * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        margin = result.financials.net_margin

        q_roe = 0.0
        if roe is not None:
            # Higher ROE is better (logarithmic to reward high-quality outliers)
            q_roe = _log_score(max(0.1, roe + 50), 30.0, -10.0) # Handle negative ROE
        
        q_debt = 0.0
        if debt_equity is not None:
            # Lower debt-to-equity is better
            q_debt = _linear_score(debt_equity, 0.2, 1.5)
        else:
            q_debt = 50.0

        q_margin = 0.0
        if margin is not None:
            q_margin = _linear_score(margin, 10.0, -5.0)
        else:
            q_margin = 50.0

        return (q_roe * 0.4 + q_debt * 0.3 + q_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on ROE and margin stability."""
        roe = result.financials.roe
        rev_growth = None # Not directly available in fields, but we can use ROE as proxy for growth efficiency
        
        # In this data model, growth is often reflected in ROE and PEG
        # We use a combination of profitability and the ability to generate returns
        g_score = 0.0
        if roe is not None:
            # High ROE often signals growth-capable companies in these datasets
            g_score = _linear_score(roe, 15.0, -5.0)
        else:
            g_score = 50.0

        # Check for cash flow strength as a growth-enabler
        ocf = result.financials.operating_cash_flow
        if ocf is not None and ocf > 0:
            # Using a placeholder logic since we don't have growth rates, 
            # but high OCF relative to net income is a quality-growth hybrid.
            g_score = min(100.0, g_score + 10.0)

        return max(0.0, min(100.0, g_score))

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite_score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results