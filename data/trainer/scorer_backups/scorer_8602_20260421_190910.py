"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Reduced from 0.65 to allow more room for quality/growth
    "quality": 0.25,   # Increased to reward stable profitability and low leverage
    "growth": 0.25,    # Increased to capture expansion potential
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
            # Ensure a small epsilon to prevent zero-out in geometric mean if score is 0
            normalized_val = max(0.001, score_val / 100.0)
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The product of powers is (S1^w1 * S2^w2 ...). 
            # To get the weighted geometric mean, we need to normalize by total weight.
            # Mathematically: (S1^w1 * S2^w2 ...)^(1/total_weight)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Calculate value score based on PE and PB ratios."""
        # Use forward PE if available, otherwise trailing PE
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        
        # If PE is negative, we treat it as a very high (bad) value or neutral.
        # For this purpose, we'll cap low PE to avoid math errors in log/linear scaling.
        pe_val = pe if (pe is not None and pe > 0) else 100.0
        pb_val = pb if (pb is not None and pb > 0) else 1.0

        # Scoring: lower PE/PB is better
        # We map typical ranges (e.g., PE 5 to 40) to scores
        score_pe = _linear_score(pe_val, best=5.0, worst=40.0)
        score_pb = _linear_score(pb_val, best=1.0, worst=10.0)
        
        # Add dividend yield as a value booster
        dy = result.valuation.dividend_yield if result.valuation.dividend_yield is not None else 0.0
        score_dy = _linear_score(dy, best=5.0, worst=0.0)

        return (score_pe * 0.4 + score_pb * 0.4 + score_dy * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and leverage."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt_equity = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        
        # ROE: higher is better (target 15-20% as best)
        score_roe = _linear_score(roe, best=0.20, worst=-0.10)
        # Leverage: lower is better (target < 0.5 as best)
        score_leverage = _linear_score(debt_equity, best=0.3, worst=1.5)
        # Margin: higher is better
        score_margin = _linear_score(margin, best=0.25, worst=-0.05)
        
        return (score_roe * 0.4 + score_leverage * 0.3 + score_margin * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG: lower is better (target 1.0 as best)
        # Handle potential negative/zero PEG via clamping in _linear_score logic or manual check
        peg_val = peg if (peg is not None and peg > 0) else 5.0
        score_peg = _linear_score(peg_val, best=0.5, worst=3.0)
        
        # ROE as a proxy for growth/return potential
        score_roe = _linear_score(roe, best=0.25, worst=-0.1)
        
        return (score_peg * 0.5 + score_roe * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
            
        return sorted_results