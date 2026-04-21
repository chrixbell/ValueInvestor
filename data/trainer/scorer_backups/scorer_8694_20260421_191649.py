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
    "quality": 0.25,   # Increased to reward fundamental stability
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent math domain errors with 0.0 scores in geometric mean
            # but allow the zero to pull the score down significantly.
            normalized_score = max(0.0, score_val) / 100.0
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The geometric mean of normalized scores scaled back to [0, 100]
            # If any score is 0, the product is 0.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use forward PE if available, else trailing PE.
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        # We use a mixture of valuation metrics
        # If PE is negative (loss-making), it's treated as a very high/bad value in linear terms,
        # but we need to handle it. For simplicity, we use a thresholding approach.
        
        v_scores = []
        if pe is not None and pe > 0:
            # A reasonable range for PE in China/HK markets
            v_scores.append(_linear_score(pe, 40.0, 2.0))
        if pb is not None and pb > 0:
            v_scores.append(_linear_score(pb, 5.0, 0.5))
        if ps is not None and ps > 0:
            v_scores.append(_linear_score(ps, 3.0, 0.5))
        if result.valuation.dividend_yield is not None:
            # Dividend yield: higher is better (inverse of PE/PB logic)
            v_scores.append(_linear_score(result.valuation.dividend_yield, 10.0, 0.0))

        if not v_scores:
            return 50.0
        return sum(v_scores) / len(v_scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage."""
        q_scores = []
        
        # ROE is a primary quality metric
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        # Map ROE (assuming % as decimal or integer) to a score. 
        # If ROE is 0.15 (15%), we treat it as 15.
        roe_val = roe * 100 if abs(roe) <= 1.0 else roe
        q_scores.append(_linear_score(roe_val, 25.0, -5.0))

        # Margin
        net_margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        nm_val = net_margin * 100 if abs(net_margin) <= 1.0 else net_margin
        q_scores.append(_linear_score(nm_val, 20.0, -10.0))

        # Leverage
        d_e = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        if d_e > 0:
            q_scores.append(_linear_score(d_e, 1.0, 3.0)) # Lower debt is better
        else:
            q_scores.append(100.0)

        return sum(q_scores) / len(q_scores)

    def _growth_score(self, result: Screening_Result) -> float:
        # Note: The prompt implies we need to implement these if they were missing, 
        # but the provided source had them as placeholders or implied.
        # Since I must provide the COMPLETE file, and the original didn't have them 
        # fully implemented in the snippet provided (it was just the class structure),
        # I will implement them to ensure the code is valid.
        pass

    # Wait, looking at the provided 'Current scorer.py', the methods _value_score, 
    # _quality_score, and _growth_score were NOT defined in the class body. 
    # I must include them to make it a valid, working file.

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_forward if result.valuation.pe_forward is not None else result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        dy = result.valuation.dividend_yield

        scores = []
        if pe is not None and pe > 0: scores.append(_linear_score(pe, 35.0, 2.0))
        else: scores.append(50.0)
            
        if pb is not None and pb > 0: scores.append(_linear_score(pb, 4.0, 0.5))
        else: scores.append(50.0)

        if ps is not None and ps > 0: scores.append(_linear_score(ps, 3.0, 0.5))
        else: scores.append(50.0)

        if dy is not None: scores.append(_linear_score(dy, 8.0, 0.0))
        else: scores.append(50.0)

        return sum(scores) / len(scores)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        # Handle cases where ROE might be passed as 0.15 or 15.0
        roe_adj = roe * 100 if (roe is not None and 0 < abs(roe) < 1.5) else roe
        
        net_margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        nm_adj = net_margin * 100 if (net_margin is not None and 0 < abs(net_margin) < 1.5) else net_margin
        
        scores = []
        scores.append(_linear_score(roe_adj, 20.0, -5.0))
        scores.append(_linear_score(nm_adj, 15.0, -5.0))
        
        de = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 0.5
        if de > 0:
            scores.append(_linear_score(de, 0.5, 2.0))
        else:
            scores.append(100.0)

        return sum(scores) / len(scores)

    def _growth_score(self, result: ScreeningResult) -> float:
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else None
        # Growth is often hard to capture without revenue growth, but we use PEG and ROE.
        # High ROE is a proxy for efficient growth.
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        roe_adj = roe * 100 if (roe is not None and 0 < abs(roe) < 1.5) else roe
        
        scores = []
        if peg is not None and peg > 0:
            # Lower PEG is better for growth-at-a-reasonable-price
            scores.append(_linear_score(peg, 2.0, 0.1))
        else:
            scores.append(50.0)
            
        # Add ROE as a growth-engine factor
        scores.append(_linear_score(roe_adj, 20.0, -5.0))

        return sum(scores) / len(scores)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        for i, r in enumerate(results):
            r.rank = i + 1
        return results