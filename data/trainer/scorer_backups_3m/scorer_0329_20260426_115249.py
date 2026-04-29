"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights – shifted emphasis from pure value to quality
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.35,
    "quality": 0.30,
    "growth": 0.15,
    "momentum": 0.00,
    "size": 0.10,
    "value_quality_interaction": 0.10,
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

    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)

        if log_best == log_worst:
            return 50.0

        score = (log_value - log_worst) / (log_best - log_worst) * 100.0
        return max(0.0, min(100.0, score))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _sigmoid_score(value: float, midpoint: float, steepness: float) -> float:
    """Return a score in [0, 100] using a sigmoid (logistic) function.

    The sigmoid is centered at *midpoint* and has slope *steepness*.
    Values far above midpoint approach 100, far below approach 0.
    """
    # Clamp exponent to avoid overflow
    exponent = steepness * (value - midpoint)
    if exponent > 700:
        return 100.0
    if exponent < -700:
        return 0.0
    sig = 1.0 / (1.0 + math.exp(-exponent))
    return sig * 100.0


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, growth, and size dimensions."""

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

        # Size score – new
        size_score = self._size_score(result)

        # Value-quality interaction score
        vq_score = self._value_quality_interaction_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
            "size": size_score,
            "value_quality_interaction": vq_score,
        }

        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Weighted sum aggregation
        composite_score = 0.0
        total_weight = 0.0

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            composite_score += weight * score_val

        if total_weight > 0:
            composite_score /= total_weight
        else:
            composite_score = 50.0

        result.composite_score = composite_score
        return result

    def _value_score(self, result: ScreeningResult) -> float:
        """Lower PE/PB/PS → higher score, using geometric mean of available components."""
        pe = result.valuation.pe_ratio
        pb = result.valuation.pb_ratio
        ps = result.valuation.ps_ratio

        pe_score = None
        pb_score = None
        ps_score = None

        # PE score
        if pe is not None and pe > 0:
            pe_score = _linear_score(1.0/pe, 1.0/50.0, 1.0/5.0)

        # PB score
        if pb is not None and pb > 0:
            pb_score = _linear_score(1.0/pb, 1.0/10.0, 1.0/1.0)

        # PS score
        if ps is not None and ps > 0:
            ps_score = _linear_score(1.0/ps, 1.0/0.5, 1.0/10.0)

        scores = [s for s in (pe_score, pb_score, ps_score) if s is not None]

        if not scores:
            return 50.0

        product = 1.0
        for s in scores:
            product *= s
        return math.pow(product, 1.0 / len(scores))

    def _quality_score(self, result: ScreeningResult) -> float:
        """Higher ROE, net margin, FCF margin, asset turnover, lower leverage,
        higher OCF/NI, and healthy current ratio → higher score.
        Uses harmonic mean of component scores to heavily penalize any single weak metric.
        """
        roe = result.financials.roe if result.financials.roe is not None else 0.0
        margin = result.financials.net_margin if result.financials.net_margin is not None else 0.0

        roe_score = _sigmoid_score(roe, midpoint=12.0, steepness=0.12)
        margin_score = _sigmoid_score(margin, midpoint=5.0, steepness=0.4)

        # Free cash flow margin score
        fcf = result.financials.free_cash_flow
        revenue = result.financials.revenue
        fcf_margin_score = 50.0
        if fcf is not None and revenue is not None and revenue > 0:
            fcf_margin = fcf / revenue
            fcf_margin_score = _linear_score(fcf_margin, 0.20, -0.10)
        elif fcf is not None and fcf > 0:
            fcf_margin_score = 60.0

        # Asset turnover score
        turnover_score = 50.0
        total_assets = result.financials.total_assets
        if revenue is not None and total_assets is not None and total_assets > 0:
            turnover = revenue / total_assets
            turnover_score = _linear_score(turnover, 2.0, 0.2)
        elif revenue is not None and revenue > 0:
            turnover_score = 55.0

        # Leverage score
        dte = result.financials.debt_to_equity
        leverage_score = 50.0
        if dte is not None and dte >= 0:
            raw = _linear_score(dte, 0.0, 3.0)
            leverage_score = 100.0 - raw
        elif dte is not None and dte < 0:
            leverage_score = 50.0

        # Operating cash flow / net income (earnings quality)
        ocf = result.financials.operating_cash_flow
        ni = result.financials.net_income
        ocf_ni_score = 50.0
        if ocf is not None and ni is not None and ni > 0:
            ratio = ocf / ni
            ocf_ni_score = _sigmoid_score(ratio, midpoint=1.0, steepness=3.0)

        # ----- New component: current ratio (liquidity) -----
        cr = result.financials.current_ratio
        cr_score = 50.0
        if cr is not None:
            # A current ratio between 1.5 and 3 is considered healthy
            # Map to [0,100] with best=2.0, worst=0.5 (below 0.5 is very risky)
            cr_score = _linear_score(cr, best=2.0, worst=0.5)

        # Combine component scores using harmonic mean
        components = [
            roe_score,
            margin_score,
            fcf_margin_score,
            turnover_score,
            leverage_score,
            ocf_ni_score,
            cr_score,
        ]
        epsilon = 1e-6
        sum_inv = 0.0
        count = 0
        for s in components:
            sum_inv += 1.0 / (s + epsilon)
            count += 1
        if count == 0:
            return 50.0
        harmonic_mean = count / sum_inv - epsilon
        return max(0.0, min(100.0, harmonic_mean))

    def _growth_score(self, result: ScreeningResult) -> float:
        """Higher ROE, lower PEG → higher score.
        Note: ROE removed to reduce redundancy with quality sub-score.
        Growth now relies solely on PEG ratio.
        """
        peg = result.valuation.peg_ratio

        if peg is not None and peg > 0:
            peg_score = _linear_score(1.0/peg, 1.0/0.5, 1.0/3.0)
        else:
            peg_score = 50.0

        return peg_score

    def _size_score(self, result: ScreeningResult) -> float:
        """Smaller market cap → higher score (small-cap tilt)."""
        mcap = result.valuation.market_cap_rmb
        if mcap is None or mcap <= 0:
            return 50.0

        log_mcap = math.log10(mcap)
        log_best = math.log10(1e8)   # 100 million
        log_worst = math.log10(1e13) # 10 trillion
        score = _linear_score(log_mcap, log_best, log_worst)
        return 100.0 - score

    def _value_quality_interaction_score(self, result: ScreeningResult) -> float:
        """Score stocks that are both cheap (low PE) and profitable (high ROE)."""
        pe = result.valuation.pe_ratio
        roe = result.financials.roe

        earnings_yield = 0.0
        if pe is not None and pe > 0:
            earnings_yield = 1.0 / pe

        roe_val = 0.0
        if roe is not None and roe > 0:
            roe_val = roe / 100.0

        product = earnings_yield * roe_val
        return _linear_score(product, best=0.03, worst=0.0)

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Rank all results based on composite score."""
        for r in results:
            self.score(r)

        sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)

        for i, res in enumerate(sorted_results):
            res.rank = i + 1

        return sorted_results