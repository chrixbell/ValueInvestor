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
    "quality": 0.25,   # Increased to reward stability and profitability
    "growth": 0.25,    # Increased to capture expansion potential
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

        # To prevent a single zero sub-score from zeroing out the entire composite (which happens 
        # in a pure geometric mean), we use an epsilon-offset for the normalization.
        # This ensures that if a stock has one bad metric, it's penalized but not eliminated.
        epsilon = 1e-6
        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Normalize score to [0, 1] range. Using epsilon to ensure zero-scores don't kill the product.
            normalized_val = (score_val / 100.0) + epsilon
            product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # Re-scale back to [0, 100]
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
            # If epsilon was added, we pull it back slightly to keep range clean
            composite_score = max(0.0, min(100.0, composite_score))
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        v = result.valuation
        if not v or (v.pe_ratio is None and v.pb_ratio is None and v.ps_ratio is None):
            return 50.0

        # Prioritize PE, then PB, then PS
        if v.pe_ratio is not None and v.pe_ratio > 0:
            return _linear_score(v.pe_ratio, 15.0, 60.0)
        elif v.pb_ratio is not None and v.pb_ratio > 0:
            return _linear_score(v.pb_ratio, 1.5, 6.0)
        elif v.ps_ratio is not None and v.ps_ratio > 0:
            return _linear_score(v.ps_ratio, 1.0, 5.0)
        
        return 50.0

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/Margin and lower leverage is better."""
        f = result.financials
        if not f:
            return 50.0

        # Quality component: ROE and Net Margin
        roe = f.roe if f.roe is not None else 0.0
        margin = f.net_margin if f.net_margin is not None else 0.0
        
        # Leverage component: Debt to Equity
        debt_to_equity = f.debt_to_equity if f.debt_to_equity is not None else 0.5

        # Score for ROE (target ~15-25%)
        roe_s = _linear_score(roe, 20.0, 5.0) if roe > 0 else 0.0
        # Score for Margin (target ~15%)
        margin_s = _linear_score(margin, 0.15, 0.02) if margin > 0 else 0.0
        # Score for Leverage (lower is better)
        leverage_s = _linear_score(debt_to_equity, 0.5, 2.0) if debt_to_equity > 0 else 100.0

        return (roe_s * 0.4) + (margin_s * 0.4) + (leverage_s * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG is better."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        roe = f.roe if f.roe is not None else 0.0
        peg = v.peg_ratio if (v.peg_ratio is not None and v.peg_ratio > 0) else None

        # Growth is often correlated with ROE, but we check PEG for valuation-adjusted growth
        if peg is not None:
            # A PEG of 1.0 is fair, < 1.0 is great.
            peg_s = _linear_score(peg, 0.5, 3.0)
            # We blend PEG with ROE to ensure we don't just pick low-growth value traps
            return (peg_s * 0.7) + (_linear_score(roe, 15.0, 5.0) * 0.3 if roe > 0 else 0.0)
        else:
            return _linear_score(roe, 15.0, 5.0) if roe > 0 else 0.0

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Sorts results by composite score and assigns ranks."""
        for r in results:
            self.score(r)
        
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results