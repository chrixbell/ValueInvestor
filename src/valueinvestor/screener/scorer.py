"""
Multi-factor scoring and ranking for screening results.
Clean restart — simple arithmetic weighted mean, independent score/rank.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.45,
    "quality": 0.30,
    "growth": 0.15,
    "momentum": 0.10,
}


def _linear_score(value: float, best: float, worst: float) -> float:
    """Map value to [0, 100] linearly, clamped."""
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
    """Map ratio metrics to [0, 100] using log scale."""
    if value <= 0 or best <= 0 or worst <= 0:
        return 0.0
    try:
        log_value = math.log(value)
        log_best = math.log(best)
        log_worst = math.log(worst)
    except (OverflowError, ValueError):
        return 0.0
    if log_best == log_worst:
        return 50.0
    score = (log_value - log_worst) / (log_best - log_worst) * 100.0
    return max(0.0, min(100.0, score))


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, growth, and momentum."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: ScreeningResult) -> ScreeningResult:
        """Compute all sub-scores and composite. Independent per-stock."""
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)
        momentum_score = self._momentum_score(result)

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": momentum_score,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for key, score_val in scores.items():
            w = self.weights.get(key, 0.0)
            if w > 0:
                total_weight += w
                weighted_sum += score_val * w

        result.composite_score = (
            weighted_sum / total_weight if total_weight > 0 else 50.0
        )
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Score all stocks, sort by composite descending, assign ranks."""
        for r in results:
            self.score(r)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # Sub-score methods
    # ------------------------------------------------------------------

    @staticmethod
    def _value_score(result: ScreeningResult) -> float:
        """Value: lower PE, PB, PS, EV/EBITDA; higher earnings yield, dividend yield."""
        weighted: List[Tuple[float, float]] = []

        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            weighted.append((_log_score(pe, best=6.0, worst=50.0), 0.20))

        pe_fwd = result.valuation.pe_forward
        if pe_fwd is not None and pe_fwd > 0:
            weighted.append((_log_score(pe_fwd, best=5.0, worst=40.0), 0.15))

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted.append((_log_score(pb, best=0.8, worst=5.0), 0.15))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted.append((_log_score(ps, best=0.3, worst=4.0), 0.10))

        ev_ebitda = result.valuation.ev_to_ebitda
        if ev_ebitda is not None and ev_ebitda > 0:
            weighted.append((_log_score(ev_ebitda, best=4.0, worst=25.0), 0.10))
        ni = result.financials.net_income
        mcap = result.valuation.market_cap_rmb
        if ni is not None and mcap is not None and ni > 0 and mcap > 0:
            ey = ni / mcap
            weighted.append((_log_score(ey, best=0.10, worst=0.01), 0.15))
        dy = result.valuation.dividend_yield
        if dy is not None and dy >= 0:
            weighted.append((_linear_score(dy, best=0.04, worst=0.0), 0.10))

        dte = result.financials.debt_to_equity
        if dte is not None:
            if dte > 0:
                weighted.append((_log_score(dte, best=0.1, worst=5.0), 0.05))
            else:
                weighted.append((100.0, 0.05))

        if not weighted:
            return 50.0
        total_w = sum(w for _, w in weighted)
        return sum(s * w for s, w in weighted) / total_w if total_w > 0 else 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Quality: higher ROE, gross margin, ROA; lower leverage."""
        weighted: List[Tuple[float, float]] = []

        roe = result.financials.roe
        if roe is not None and roe > 0:
            weighted.append((_linear_score(roe, best=0.25, worst=0.03), 0.35))

        gm = result.financials.gross_margin
        if gm is not None and gm > 0:
            weighted.append((_linear_score(gm, best=0.60, worst=0.05), 0.30))

        dte = result.financials.debt_to_equity
        if dte is not None:
            if dte > 0:
                weighted.append((_log_score(dte, best=0.1, worst=3.0), 0.20))
            else:
                weighted.append((100.0, 0.20))

        roa = result.financials.roa
        if roa is not None and roa > 0:
            weighted.append((_linear_score(roa, best=0.15, worst=0.0), 0.15))

        if not weighted:
            return 50.0
        total_w = sum(w for _, w in weighted)
        return sum(s * w for s, w in weighted) / total_w if total_w > 0 else 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Growth: lower PEG, better forward PE trajectory."""
        weighted: List[Tuple[float, float]] = []

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            weighted.append((_log_score(peg, best=0.5, worst=2.5), 0.55))

        pe = result.valuation.pe_ratio
        pe_fwd = result.valuation.pe_forward
        if pe is not None and pe_fwd is not None and pe > 0 and pe_fwd > 0:
            fw_improve = pe / pe_fwd
            weighted.append((_linear_score(fw_improve, best=2.0, worst=0.5), 0.45))

        if not weighted:
            return 50.0
        total_w = sum(w for _, w in weighted)
        return sum(s * w for s, w in weighted) / total_w if total_w > 0 else 50.0

    @staticmethod
    def _momentum_score(result: ScreeningResult) -> float:
        """Momentum: placeholder — no price trend data available."""
        return 50.0
