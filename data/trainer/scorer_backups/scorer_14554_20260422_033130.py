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
    "quality": 0.25,   # Increased to capture fundamental stability
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

        # We use a weighted arithmetic mean for the final combination to prevent 
        # one zero-score factor from zeroing out an otherwise excellent stock.
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
        """Calculate value score based on PE and PB."""
        v = result.valuation
        if not v:
            return 50.0

        # Using PE and PB as primary value drivers. 
        # We use a small epsilon to avoid division by zero/log issues if needed, 
        # though we use linear interpolation here.
        pe = v.pe_ratio if (v.pe_ratio is not None and v.pe_ratio > 0) else None
        pb = v.pb_ratio if (v.pb_ratio is not None and v.pb_ratio > 0) else None
        
        # Fallback to dividend yield if PE/PB are missing or problematic
        dy = v.dividend_yield if (v.dividend_yield is not None and v.dividend_yield > 0) else None

        # Scoring logic: lower PE/PB is better.
        # We'll use a simple multi-factor combination for the sub-score itself.
        scores = []
        if pe:
            # Typically PE between 5 and 30 is a reasonable range for 'value'
            scores.append(_linear_score(pe, 5, 40)) # Low PE is better
            # Wait, _linear_score: (value - worst) / (best - worst). 
            # To make lower PE better, we swap: best=5, worst=40. 
            # But the function logic is (val - worst)/(best-worst). 
            # Let's use (worst - val) / (worst - best) logic? 
            # Actually, let's just define:
            score_pe = (40 - pe) / (40 - 5) * 100
            scores.append(max(0.0, min(100.0, score_pe)))
        
        if pb:
            score_pb = (5 - pb) / (5 - 0.1) * 100
            scores.append(max(0.0, min(100.0, score_pb)))
        
        if dy:
            score_dy = min(100.0, dy * 20) # Assume 5% yield is a good score
            scores.append(score_dy)

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Calculate quality score based on ROE and Debt/Equity."""
        f = result.financials
        if not f:
            return 50.0

        scores = []
        # ROE is a primary quality metric
        roe = f.roe if f.roe is not None else None
        if roe is not None:
            # Map ROE (e.g., -20 to 40) to score
            score_roe = (roe - (-10)) / (30 - (-10)) * 100
            scores.append(max(0.0, min(100.0, score_roe)))
        
        # Debt to Equity
        de = f.debt_to_equity if f.debt_to_equity is not None else None
        if de is not None:
            # Lower debt is better. Assume 1.0 (100%) is worst-neutral, 0 is best.
            score_de = (1.0 - de) / 1.0 * 100
            scores.append(max(0.0, min(100.0, score_de)))

        # Margin
        margin = f.net_margin if f.net_margin is not None else None
        if margin is not None:
            score_margin = (margin - (-0.1)) / (0.3 - (-0.1)) * 100
            scores.append(max(0.0, min(100.0, score_margin)))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Calculate growth score based on PEG and ROE."""
        v = result.valuation
        f = result.financials
        if not v or not f:
            return 50.0

        scores = []
        peg = v.peg_ratio if (v.peg_ratio is not None and v.peg_ratio > 0) else None
        roe = f.roe if f.roe is not None else None

        if peg:
            # Lower PEG is better (growth relative to value)
            score_peg = (2.0 - peg) / 2.0 * 100
            scores.append(max(0.0, min(100.0, score_peg)))
        
        if roe:
            # High ROE is a proxy for efficient growth
            score_roe = (roe - (-0.1)) / (0.4 - (-0.1)) * 100
            scores.append(max(0.0, min(100.0, score_roe)))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        # Sort descending: highest score is rank 1
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, res in enumerate(results):
            res.rank = i + 1
        return results