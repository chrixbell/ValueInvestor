"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,     # Slightly reduced to allow more weight for quality/growth
    "quality": 0.25,   # Increased to penalize poor balance sheets/profitability
    "growth": 0.25,    # Increased to capture expansionary potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best:float, worst:float) -> float:
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

        total_weight = sum(self.weights.get(k, 0.0) for k in weighted_scores_to_process)
        
        if total_weight == 0:
            result.composite_score = 50.0
            return result

        # Using a weighted arithmetic mean for the sub-scores to prevent 
        # a single zero score (from one bad metric) from destroying the entire composite.
        weighted_sum = 0.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            weighted_sum += score_val * weight

        result.composite_score = weighted_sum / total_weight
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB and high dividend yield are better."""
        v = result.valuation
        f = result.financials
        
        # Primary valuation metric: PE or PB if PE is unavailable/negative
        if v.pe_ratio and v.pe_ratio > 0:
            val = v.pe_ratio
            best, worst = 5.0, 40.0
            score = _linear_score(val, worst, best) # Inverted: lower is better
        elif v.pb_ratio and v.pb_ratio > 0:
            val = v.pb_ratio
            best, worst = 1.0, 6.0
            score = _linear_score(val, worst, best)
        else:
            score = 50.0

        # Add dividend yield component (higher is better)
        if v.dividend_yield and v.dividend_yield > 0:
            div_score = _linear_score(v.dividend_yield, 7.0, 0.0) # Using custom logic for inversion
            # Re-map div_score correctly: we want higher yield -> higher score.
            # Let's just use a simpler approach:
            div_score = min(100.0, (v.dividend_yield / 7.0) * 100.0) if v.dividend_yield < 7.0 else 100.0
            # Actually, let's stick to a simpler weight-based logic:
            score = (score * 0.7) + (min(100.0, v.dividend_yield * 20) * 0.3)
        
        return score

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and margin, lower leverage."""
        f = result.financials
        
        # ROE component
        roe_score = 0.0
        if f.roe is not None:
            # Normalize ROE (assuming 20% is great, -10% is bad)
            roe_score = _linear_score(f.roe, 20.0, -10.0)
        
        # Margin component
        margin_score = 0.0
        if f.net_margin is not None:
            margin_score = _linear_score(f.net_margin, 15.0, -5.0)
            
        # Leverage component (lower is better)
        lev_score = 50.0
        if f.debt_to_equity is not None and f.debt_to_equity > 0:
            lev_score = _linear_score(f.debt_to_equity, 0.5, 2.0)
            
        # Combine: ROE and Margins are quality indicators, Leverage is a risk indicator.
        return (roe_score * 0.4) + (margin_score * 0.4) + (lev_score * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE and lower PEG."""
        v = result.valuation
        f = result.financials
        
        # Growth is often captured by the combination of ROE and PEG
        roe_score = 0.0
        if f.roe is not None:
            roe_score = _linear_score(f.roe, 15.0, -5.0)
            
        peg_score = 50.0
        if v.peg_ratio and v.peg_ratio > 0:
            # Lower PEG is better for growth-at-reasonable-price
            peg_score = _linear_score(v.peg_ratio, 0.5, 3.0)
        elif v.pe_ratio and v.pe_ratio > 0:
            # If no PEG, fallback to PE-based growth proxy
            peg_score = _linear_score(v.pe_ratio, 10.0, 30.0)

        return (roe_score * 0.5) + (peg_score * 0.5)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
            
        return results