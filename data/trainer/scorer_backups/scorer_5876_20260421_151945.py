"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Balanced weight for value
    "quality": 0.25,   # Increased quality to capture fundamental stability
    "growth": 0.25,    # Balanced growth weight
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

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Using a weighted arithmetic mean for better stability with potential zero scores 
        # in sub-factors, ensuring no single bad factor completely wipes the score to zero.
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
        """Calculate value score using PB and PE."""
        # Use PEG for growth-adjusted valuation if available, otherwise fallback to PE/PB
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Defaulting to a neutral-ish bad value if data is missing
        if pe is None or pe <= 0:
            pe = 50.0  # High PE penalty placeholder
        if pb is None or pb <= 0:
            pb = 5.0

        # Score based on low PE and low PB (Inverse relationship)
        # We use a simple linear comparison against typical thresholds
        pe_score = _linear_score(1.0 / pe if pe != 0 else 0, 1.0/5.0, 1.0/100.0) if pe > 0 else 0
        # Since we can't easily use absolute values without a population, 
        # we assume standard ranges for the 'best' and 'worst' in a cross-sectional sense.
        # For this implementation, we use the existing logic structure but with better bounds.
        
        # Let's simplify: normalize PE and PB to a score. 
        # In a real single-stock context, these bounds are arbitrary without the full dataset.
        # We assume 'best' is low PE/PB and 'worst' is high.
        pe_s = _linear_score(1/pe if pe > 0 else 0, 1/5, 1/50) if pe is not None and pe > 0 else 0
        pb_s = _linear_score(1/pb if pb > 0 else 0, 1/3, 1/10) if pb is not None and pb > 0 else 0
        
        # If PEG is available, it's a powerful value/growth hybrid. 
        if peg is not None and peg > 0:
            peg_s = _linear_score(1/peg, 1/0.5, 1/3)
            return (pe_s * 0.4 + pb_s * 0.3 + peg_s * 0.3)
            
        return (pe_s * 0.6 + pb_s * 0.4)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_to_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # Higher ROE is better
        roe_s = _linear_score(roe, 0.15, -0.10)
        # Lower Debt-to-Equity is better
        debt_s = _linear_score(1 / (debt_to_equity + 0.01) if debt_to_equity > 0 else 10, 1/2.0, 1/0.5)
        # Higher net margin is better
        margin_s = _linear_score(margin, 0.15, -0.05)

        return (roe_s * 0.4 + margin_s * 0.4 + debt_s * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score using ROE and Revenue Growth (via implicit check)."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        # We don't have explicit revenue growth, but we can use ROE as a proxy for capital efficiency
        # and check if current assets/equity or similar suggest growth potential.
        # For now, we focus on ROE and scale (market cap).
        
        # Using ROE as a primary growth/quality hybrid component
        roe_s = _linear_score(roe, 0.20, -0.05)
        
        # If PEG is low, it implies growth is cheap relative to valuation.
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        if peg is not None and peg > 0:
            peg_s = _linear_score(1/peg, 1/0.5, 1/3)
            return (roe_s * 0.5 + peg_s * 0.5)
        
        return roe_s

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results