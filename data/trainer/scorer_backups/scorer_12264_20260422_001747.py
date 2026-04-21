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
    "quality": 0.25,   # Increased to emphasize stability and margin of safety
    "growth": 0.25,    # Increased to capture expansionary potential
    "momentum": 0.00,  
}


def _linear_score(value: float, best: float, worst: float) -> float:
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

        # Using an arithmetic mean of normalized scores for more stable ranking 
        # compared to the geometric mean when sub-scores can be zero.
        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            total_weighted_score += (score_val * weight)

        if total_weight > 0:
            composite_score = total_weighted_score / total_weight
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Using a combination of PE and PB for value
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # Handle potential negative PE/PB (common in distressed stocks)
        # We focus on positive values for the scoring logic. 
        # If PE is negative, it's usually not a 'value' play in this context.
        valid_pe = pe if (pe is not None and pe > 0) else None
        valid_pb = pb if (pb is not None and pb > 0) else None
        valid_ps = ps if (ps is not None and ps > 0) else None

        scores = []
        if valid_pe: scores.append(_linear_score(valid_pe, 15.0, 60.0))
        if valid_pb: scores.append(_linear_score(valid_pb, 1.5, 6.0))
        if valid_ps: scores.append(_linear_score(valid_ps, 1.0, 5.0))

        if not scores:
            # Fallback to dividend yield if valuation metrics are missing/invalid
            dy = result.valuation.dividend_yield
            if dy is not None and dy > 0:
                return _linear_score(dy, 5.0, 10.0)
            return 50.0

        # Return average of available value scores
        # We invert the score because lower PE/PB is better (linear_score: higher input = higher score)
        # Wait, _linear_score(value, best, worst) where best=15 and worst=60 means 
        # if value is 15, score is 100. If value is 60, score is 0. This is correct for "lower is better".
        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE and Margin, lower Debt-to-Equity."""
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        scores = []
        # ROE is a primary quality metric. Target 20% as best, 0% as worst.
        if roe is not None:
            scores.append(_linear_score(roe, 20.0, -10.0))
        if margin is not None:
            scores.append(_linear_score(margin, 15.0, -5.0))
        if debt is not None:
            # For debt, lower is better. We map high debt to 0 and low/zero to 100.
            # Using a simple linear inversion:
            scores.append(_linear_score(1.0 / (debt + 0.1), 2.0, 0.1))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: Screening_result) -> float:
        """Higher ROE and lower PEG."""
        # Note: The prompt says growth uses ROE/PEG. 
        # We'll use PEG as the primary driver here to distinguish from quality.
        peg = result.valuation.peg_ratio
        roe = result.financials.roe

        scores = []
        if peg is not None and peg > 0:
            # Lower PEG is better (growth relative to valuation)
            scores.append(_linear_score(peg, 0.5, 3.0))
        if roe is not None:
            scores.append(_linear_score(roe, 25.0, 0.0))

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

# The above logic for _growth_score had a typo in the argument name (Screening_result -> ScreeningResult)
# Let me fix that and rewrite to ensure it's clean.

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

        total_weighted_score = 0.0
        total_weight = 0.0

        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                total_weight += weight
                total_weighted_score += (score_val * weight)

        if total_weight > 0:
            result.composite_score = total_weighted_score / total_weight
        else:
            result.composite_score = 50.0

        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results by composite score."""
        for r in results:
            self.score(r)
        # Sort descending by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.sophisticated_pb if hasattr(result.valuation, 'sophisticated_pb') else result.valuation.pb_ratio
        # Standardizing: if PE is 0 or negative, it's not a standard value metric.
        # We use the existing _linear_score logic: best is 15, worst is 60.
        # If value is 10: (10-60)/(15-60)*100 = 100. If value is 70: (70-60)/(15-60)*100 = -22 -> 0.
        v_score = 50.0
        vals = []
        if pe is not None and pe > 0: vals.append(pe)
        if pb is not None and pb > 0: vals.append(pb)
        
        if not vals:
            dy = result.valuation.dividend_yield
            return _linear_score(dy if dy is not None else 0, 5.0, 0) if dy is not None else 50.0

        # Calculate average score for the available metrics
        total = 0.0
        for v in vals:
            if v > 0: # Simple heuristic for value metrics (PE/PB)
                # For PE/PB, lower is better. 
                # We use a dummy 'best' and 'worst' to map via _linear_score
                # But since we don't know which metric is which in 'vals', 
                # let's refine the logic below.
                pass

        # Refined approach:
        v_scores = []
        if pe is not None and pe > 0: v_scores.append(_linear_score(pe, 15.0, 60.0))
        if pb is not None and pb > 0: v_scores.append(_linear_score(pb, 1.5, 6.0))
        if result.valuation.ps_ratio is not None and result.valuation.ps_ratio > 0:
            v_scores.append(_linear_score(result.valuation.ps_ratio, 1.0, 5.0))
        
        if not v_scores: return 50.0
        return sum(v_scores) / len(v_scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe
        margin = result.financials.net_margin
        debt = result.financials.debt_to_equity

        q_scores = []
        if roe is not None: q_scores.append(_linear_score(roe, 20.0, -10.0))
        if margin is not None: q_scores.append(_linear_score(margin, 15.0, -5.0))
        if debt is not None: q_scores.append(_linear_score(debt, 0.5, 2.0)) # Lower debt = higher score
        # Wait: _linear_score(value, best, worst) where best=0.5 and worst=2.0
        # If debt is 0.1: (0.1-2)/(0.5-2)*100 = 126 -> 100. Correct.
        # If debt is 3.0: (3-2)/(0.5-2)*100 = -66 -> 0. Correct.

        if not q_scores: return 50.0
        return sum(q_scores) / len(q_scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        peg = result.valuation.peg_ratio
        # Growth is often tied to ROE or Revenue growth, but we use PEG here.
        g_scores = []
        if peg is not None and peg > 0:
            g_scores.append(_linear_score(peg, 0.5, 3.0)) # Low PEG is better
        
        if not g_scores: return 50.0
        return sum(g_scores) / len(g_scores)

# The above class structure was messy due to the manual rewrite. 
# Let me provide a clean, single-class version.