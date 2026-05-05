"""
Multi-factor scoring and ranking for screening results.
Simple weighted arithmetic mean across 6 factors: value, quality, growth,
momentum, synergy, and value_growth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from valueinvestor.data.models import ScreeningResult

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.40,
    "quality": 0.20,
    "growth": 0.10,
    "momentum": 0.10,
    "synergy": 0.10,
    "value_growth": 0.10,
}


def _linear_score(value: float, best: float, worst: float) -> float:
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100.0
    return max(0.0, min(100.0, score))


def _log_score(value: float, best: float, worst: float) -> float:
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
    """Score :class:`ScreeningResult` objects across value, quality, growth,
    momentum, synergy, and value_growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: ScreeningResult) -> ScreeningResult:
        result.value_score = self._value_score(result)
        result.quality_score = self._quality_score(result)
        result.growth_score = self._growth_score(result)
        result.momentum_score = self._momentum_score(result)
        result.synergy_score = min(result.value_score, result.quality_score)
        result.value_growth_score = math.sqrt(
            max(result.value_score, 0.0) * max(result.growth_score, 0.0)
        )

        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": result.momentum_score,
            "synergy": result.synergy_score,
            "value_growth": result.value_growth_score,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for k, score_val in scores.items():
            w = self.weights.get(k, 0.0)
            if w > 0:
                total_weight += w
                weighted_sum += w * score_val

        if total_weight > 0:
            result.composite_score = max(0.0, min(100.0, weighted_sum / total_weight))
        else:
            result.composite_score = 50.0
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        for r in results:
            self.score(r)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # Factor sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def _value_score(result: ScreeningResult) -> float:
        v = result.valuation
        f = result.financials
        total_w = 0.0
        weighted_sum = 0.0

        pe = v.pe_ratio
        if pe is not None and pe > 0:
            w = 0.20
            total_w += w
            weighted_sum += w * _log_score(pe, best=6.0, worst=50.0)

        pe_fwd = v.pe_forward
        if pe_fwd is not None and pe_fwd > 0:
            w = 0.15
            total_w += w
            weighted_sum += w * _log_score(pe_fwd, best=5.0, worst=40.0)

        pb = v.pb_ratio
        if pb is not None and pb > 0:
            w = 0.15
            total_w += w
            weighted_sum += w * _log_score(pb, best=0.8, worst=5.0)

        ps = v.ps_ratio
        if ps is not None and ps > 0:
            w = 0.10
            total_w += w
            weighted_sum += w * _log_score(ps, best=0.3, worst=4.0)

        ev_ebitda = v.ev_to_ebitda
        if ev_ebitda is not None and ev_ebitda > 0:
            w = 0.10
            total_w += w
            weighted_sum += w * _log_score(ev_ebitda, best=4.0, worst=25.0)

        ni = f.net_income
        mcap = v.market_cap_rmb
        if ni is not None and mcap is not None and mcap > 0 and ni > 0:
            ey = ni / mcap
            w = 0.15
            total_w += w
            weighted_sum += w * _linear_score(ey, best=0.10, worst=0.01)

        div_yield = v.dividend_yield
        if div_yield is not None and div_yield >= 0:
            w = 0.10
            total_w += w
            weighted_sum += w * _linear_score(div_yield, best=0.04, worst=0.0)

        dte = f.debt_to_equity
        if dte is not None and dte > 0:
            w = 0.05
            total_w += w
            weighted_sum += w * _log_score(dte, best=0.1, worst=5.0)

        return weighted_sum / total_w if total_w > 0 else 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        f = result.financials
        total_w = 0.0
        weighted_sum = 0.0

        roe = f.roe
        if roe is not None and roe > 0:
            w = 0.35
            total_w += w
            weighted_sum += w * _linear_score(roe, best=0.25, worst=0.03)

        gm = f.gross_margin
        if gm is not None and gm > 0:
            w = 0.30
            total_w += w
            weighted_sum += w * _linear_score(gm, best=0.60, worst=0.05)

        dte = f.debt_to_equity
        if dte is not None and dte > 0:
            w = 0.20
            total_w += w
            weighted_sum += w * _log_score(dte, best=0.1, worst=3.0)

        roa = f.roa
        if roa is not None and roa > 0:
            w = 0.15
            total_w += w
            weighted_sum += w * _linear_score(roa, best=0.15, worst=0.0)

        return weighted_sum / total_w if total_w > 0 else 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        v = result.valuation
        total_w = 0.0
        weighted_sum = 0.0

        peg = v.peg_ratio
        if peg is not None and peg > 0:
            w = 0.55
            total_w += w
            weighted_sum += w * _log_score(peg, best=0.5, worst=2.5)

        pe = v.pe_ratio
        pe_fwd = v.pe_forward
        if pe is not None and pe_fwd is not None and pe > 0 and pe_fwd > 0:
            fw_improve = pe / pe_fwd
            w = 0.45
            total_w += w
            weighted_sum += w * _linear_score(fw_improve, best=2.0, worst=0.5)

        return weighted_sum / total_w if total_w > 0 else 50.0

    @staticmethod
    def _momentum_score(result: ScreeningResult) -> float:
        return 50.0
