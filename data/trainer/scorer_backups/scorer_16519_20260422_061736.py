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
    "quality": 0.25,   # Increased to reward stable companies more heavily
    "growth": 0.25,    # Balanced with quality to capture sustainable growth
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

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Using an additive weighted sum approach for stability in ranking 
        # when dealing with zeros or extreme outliers in sub-scores.
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
        """Calculate value score based on PE, PB, and Dividend Yield."""
        # Factors where lower is better (PE, PB)
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        dy = result.valuation.dividend_yield

        # Fallback values for comparison if None
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        dy = dy if dy is not None else 0.0

        # We want to reward low PE, low PB, and high Dividend Yield
        # Using a simple weighted average of normalized scores
        # Note: For PE/PB, the 'best' is low, so we invert logic
        
        # Score 1: PE/PB (Lower is better)
        # We check against typical ranges. High PE is penalized.
        pe_score = _linear_score(pe, 5.0, 40.0) # 5 is best, 40 is worst
        pb_score = _linear_score(pb, 1.0, 6.0)  # 1 is best, 6 is worst
        
        # Score 2: Dividend Yield (Higher is better)
        dy_score = _linear_score(dy, 5.0, 0.0) # This is tricky with linear_score logic
        # Re-implementing dy_score for clarity:
        dy_score = max(0.0, min(100.0, (dy / 5.0) * 100.0)) if dy > 0 else 0.0
        # Actually, let's use a simpler approach for the sub-score:
        
        val_score = (100.0 - pe_score + 100.0 - pb_score) / 2.0
        # If we want to include DY:
        dy_contribution = min(100.0, dy * 20.0) # 5% yield = 100
        
        # Final value score mix: (PE/PB component) + (DY component)
        # Let's simplify to a single metric for the sub-score:
        return (val_score * 0.7) + (dy_contribution * 0.3)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt."""
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        # ROE: Higher is better (Targeting 15%+)
        roe_score = max(0.0, min(100.0, (roe + 0.2) * 100.0 / 0.4)) # Normalize: 20% ROE -> 100
        # Debt: Lower is better (Targeting < 50%)
        debt_score = max(0.0, min(100.0, (0.5 - debt) / 0.5 * 100.0)) if debt < 0.5 else 0.0
        # Margin: Higher is better
        margin_score = max(0.0, min(100.0, (margin + 0.1) * 100.0 / 0.3))

        return (roe_score * 0.4) + (debt_score * 0.3) + (margin_score * 0.3)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and Revenue/Income trends."""
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None and result.valuation.peg_ratio > 0 else 2.0
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        
        # PEG: Lower is better (1.0 is ideal)
        peg_score = max(0.0, min(100.0, (2.5 - peg) / 2.5 * 100.0))
        # ROE: Higher is better (Growth through profitability)
        roe_score = max(0.0, min(100.0, (roe + 0.1) * 100.0 / 0.3))

        return (peg_score * 0.6) + (roe_score * 0.4)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(sorted_results):
            res.rank = i + 1
            
        return sorted_results