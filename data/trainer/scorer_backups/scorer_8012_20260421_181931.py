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
    "growth": 0.25,    # Increased to capture upside potential
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
            # Use a small epsilon to prevent 0.0 from zeroing out the whole product if one factor is bad
            # but keep it sensitive to low scores.
            safe_score = max(0.001, score_val) / 100.0
            product_of_powers *= (safe_score ** weight)

        if total_weight > 0:
            # The weighted geometric mean formula (product of S^w) already accounts for normalization 
            # if the weights sum to 1.0. If they don't, we adjust by total_weight.
            # We use the identity: product(S^(w/W)) = (product(S^w))^(1/W)
            composite_score = (product_of_powers ** (1.0 / total_weight)) * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Compute value score using PE and PB ratios."""
        # Use PEG as a secondary check if available
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        peg = result.valuation.peg_ratio

        # Handle potential zero/negative PE (common in distressed stocks)
        # We use a simple heuristic: higher is worse for PE/PB. 
        # If PE is negative, it's often ignored in simple value screens or treated as 'worst'.
        # We will treat positive PE/PB as the primary metric.
        
        score_pe = 0.0
        if pe is not None and pe > 0:
            # Clamp PE between 1 and 50 for scoring range
            score_pe = _linear_score(pe, 1.0, 50.0)
            # Since lower PE is better, we invert the linear score (100 - score)
            score_pe = 100.0 - score_pe
        elif pe is not None and pe <= 0:
            # Negative PE is tricky; often means loss. Let's assume it's 'bad' value but potentially 
            # high growth. We assign a low-mid score to avoid zeroing out the whole product.
            score_pe = 10.0

        score_pb = 0.0
        if pb is not None and pb > 0:
            score_pb = 100.0 - _linear_score(pb, 0.5, 10.0)
        elif pb is not None and pb <= 0:
            score_pb = 10.0

        # Combine scores (simple average of available)
        if pe is not None and pb is not None:
            return (score_pe + score_pb) / 2.0
        return score_pe if pe is not None else score_pb

    def _quality_score(self, result: ScreeningResult) -> float:
        """Compute quality score using ROE and Debt-to-Equity."""
        roe = result.financials.roe
        debt_equity = result.financials.debt_to_equity
        
        scores = []
        if roe is not None:
            # ROE score: higher is better. 
            # Mapping typical ROE (0 to 30%) to a score.
            scores.append(_linear_score(roe, 25.0, -10.0))
        
        if debt_equity is not None:
            # Debt to Equity score: lower is better. 
            # Using a range of 0% to 150%.
            scores.append(100.0 - _linear_score(debt_equity, 150.0, 0.0))
            
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Compute growth score using ROE and PEG."""
        roe = result.financials.roe
        peg = result.valuation.peg_ratio

        scores = []
        if roe is not None:
            # ROE as a proxy for growth/efficiency
            scores.append(_linear_score(roe, 20.0, -5.0))
        
        if peg is not None and peg > 0:
            # PEG ratio: lower is better.
            scores.append(100.0 - _linear_score(peg, 3.0, 0.1))
        elif peg is not None and peg <= 0:
            # Negative PEG usually means negative earnings; hard to score.
            scores.append(50.0)

        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for res in results:
            self.score(res)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, res in enumerate(results):
            res.rank = i + 1
            
        return results