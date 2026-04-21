"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,     # Reduced slightly to allow more weight for quality/growth
    "quality": 0.25,   # Increased to capture more robust fundamental stability
    "growth": 0.30,    # Increased to capture upside potential
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

        product_of_powers = 1.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            # Use a small epsilon to prevent zero-score annihilation in geometric mean,
            # but allow low scores to penalize heavily.
            normalized_score = max(0.01, score_val / 100.0)
            product_of_powers *= (normalized_score ** weight)

        if total_weight > 0:
            # The product of powers already incorporates the weights.
            # To get back to [0, 100], we scale it.
            composite_score = product_of_powers * 100.0
        else:
            composite_score = 50.0 

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        # Use a mix of valuation metrics
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        dy = result.valuation.dividend_yield

        # Fallback for missing data
        pe = pe if pe is not None and pe > 0 else 20.0
        pb = pb if pb is not None and pb > 0 else 2.0
        ps = ps if ps is not None and ps > 0 else 2.0
        dy = dy if dy is not None else 0.0

        # Scoring logic: lower ratios are better
        # We use a simple mapping where 1.0 is the "worst" reasonable ratio and 0.1 is "best"
        # In a real scenario, these bounds would be percentile-based or industry-specific.
        score_pe = _linear_score(20/pe if pe > 0 else 100, 1, 30)
        score_pb = _linear_score(2/pb if pb > 0 else 100, 1, 5)
        score_ps = _linear_score(2/ps if ps > 0 else 100, 1, 5)
        score_dy = _linear_score(dy if dy is not None else 0, 0.05, 0.0) # higher yield is better

        # Combine metrics
        return (score_pe * 0.4 + score_pb * 0.3 + score_ps * 0.2 + (100 - score_dy) * 0.1) # Wait, dy is better when higher
        # Correction: if dy is 5%, score_dy should be high.
        # Let's re-map dy:
        return (score_pe * 0.35 + score_pb * 0.25 + score_ps * 0.20 + (min(100, dy*10) if dy is not None else 0) * 0.2)

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS is better."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        dy = result.valuation.dividend_yield

        # Normalize inputs to reasonable bounds for linear scoring
        # PE: 1-30 (1 is best, 30 is worst)
        # PB: 0.5-5 (0.5 is best, 5 is worst)
        # PS: 0.5-5 (0.5 is best, 5 is worst)
        # DY: 0-10% (10% is best, 0% is worst)

        s_pe = _linear_score(1.0, 30.0, 1.0) if pe is not None and pe > 0 else 50.0
        # Note: _linear_score(val, best, worst) -> if val is 30 and worst is 1, score is 0.
        # Re-aligning: if pe=30, best=1, worst=30 -> score 0. If pe=1, score 100.
        s_pe = _linear_score(pe if pe is not None and pe > 0 else 30, 1, 30)
        s_pb = _linear_score(pb if pb is not None and pb > 0 else 5, 0.5, 5)
        s_ps = _linear_score(ps if ps is not None and ps > 0 else 5, 0.5, 5)
        s_dy = _linear_score(dy if dy is not None else 0, 0.1, 0) # Higher dy -> higher score

        # Actually, _linear_score(value, best, worst) implementation:
        # (30-30)/(1-30)*100 = 0. (1-30)/(1-30)*100 = 100. Correct.

        return (s_pe * 0.4 + s_pb * 0.2 + s_ps * 0.2 + (s_dy if dy is not None else 0) * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE/margin, lower leverage → higher score."""
        roe = result.financials.roe if result.financials.roe is not None else 0
        net_margin = result.financials.net_margin if result.financials.net_margin is not None else 0
        debt_eq = result.financials.debt_to_equity if result.finance_to_equity is not None else 0 # check field name
        # Fixed: debt_to_equity is in financials
        debt_eq = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0
        
        # Simplified scoring
        s_roe = _linear_score(roe, 20.0, -10.0)
        s_margin = _linear_score(net_margin, 15.0, -5.0)
        s_debt = _linear_score(debt_eq, 0.5, 2.0) # Lower debt is better
        # Wait, if debt_eq=0.5 (best), score = (0.5-2)/(0.5-2)*100 = 100. Correct.
        # If debt_eq=2.0 (worst), score = (2-2)/(0.5-2)*100 = 0. Correct.

        return (s_roe * 0.4 + s_margin * 0.4 + s_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG → higher score."""
        roe = result.financials.roe if result.financials.roe is not None else 0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0
        
        # PEG: lower is better (but avoid negative/zero pegs)
        s_roe = _linear_score(roe, 25.0, -10.0)
        s_peg = _linear_score(peg if peg > 0 else 5, 0.5, 3)
        
        return (s_roe * 0.5 + s_peg * 0.5)

    # The above methods were re-written because the previous ones were placeholders or 
    # needed logical consistency. I will now provide a clean, single-class implementation.

class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute sub-scores and composite score, updating *result* in place."""
        # 1. Calculate Sub-scores
        v_s = self._value_score(result)
        q_s = self._quality_score(result)
        g_s = self._growth_score(result)

        result.value_score = v_s
        result.quality_score = q_s
        result.growth_score = g_s

        # 2. Compose Score using weighted geometric mean
        # We use (score/100) to keep it in [0, 1] range for the power calculation
        # We add a tiny epsilon to avoid log(0) issues in math
        scores = {
            "value": v_s,
            "quality": q_s,
            "growth": g_s,
            "momentum": 50.0
        }

        total_weight = 0.0
        product_of_powers = 1.0

        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                total_weight += weight
                # Clamp score to [0.01, 100] to prevent zero-out in geometric mean
                normalized_val = max(0.01, score_val / 100.0)
                product_of_powers *= (normalized_val ** weight)

        if total_weight > 0:
            # The formula (S1^w1 * S2^w2) / (total_weight) is not how geometric mean works.
            # It should be product of (S/100)^weight, then multiplied by 100.
            # e.g., if weight is 1.0, result is (S/100)^1 * 100 = S.
            # If weights sum to 1.0, product_of_powers is already the normalized score.
            result.composite_score = product_of_powers * 100.0
        else:
            result.composite_score = 50.0

        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank results based on composite score."""
        for r in results:
            self.score(r)
        return sorted(results, key=lambda x: x.composite_score, reverse=True)

    def _value_score(self, result: ScreeningResult) -> float:
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio
        dy = result.valuation.dividend_yield

        # Define bounds for normalization
        s_pe = _linear_score(pe if pe is not None and pe > 0 else 30.0, 1.0, 30.0)
        s_pb = _linear_score(pb if pb is not None and pb > 0 else 5.0, 0.5, 5.0)
        s_ps = _linear_score(ps if ps is not None and ps > 0 else 5.0, 0.5, 5.0)
        s_dy = _linear_score(dy if dy is not None else 0.0, 0.1, 0.0)

        return (s_pe * 0.4 + s_pb * 0.2 + s_ps * 0.2 + s_dy * 0.2)

    def _quality_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0
        debt = result.financials.debt_to_equity if result.financials.debt_to_equity is not None else 1.0

        s_roe = _linear_score(roe, 20.0, -10.0)
        s_margin = _linear_score(margin, 15.0, -5.0)
        s_debt = _linear_score(debt, 0.5, 2.0)

        return (s_roe * 0.4 + s_margin * 0.4 + s_debt * 0.2)

    def _growth_score(self, result: ScreeningResult) -> float:
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        peg = result.valuation.peg_ratio if result.valuation.peg_ratio is not None else 1.0

        s_roe = _linear_score(roe, 25.0, -10.0)
        s_peg = _linear_score(peg if peg > 0 else 5.0, 0.5, 3.0)

        return (s_roe * 0.5 + s_peg * 0.5)

# The class above contains the logic but needs to be properly structured for the final output.
# I will consolidate into a single valid class structure below.