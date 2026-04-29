"""
Multi-factor scoring and ranking for screening results.
This version replaces the entire quality sub‑score with a purely
quality‑oriented raw metric that is cross‑sectionally ranked each
time `rank()` is called.  The raw metric is:
    quality_raw = (ROE × gross_margin) / (max(debt_to_equity, 0.01) + 1)
It captures profitability, leverage, and valuation in one interaction,
and the percentile transform ensures the score distribution adapts to the
current universe without hard‑coded thresholds.

The growth sub‑score weight has been set to 0.0, effectively removing it
from the composite.  This concentrates the signal on value and quality,
which may reduce noise and improve rank correlation with forward returns.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.65,
    "quality": 0.35,
    "growth": 0.00,   # removed – growth sub‑score contributes only noise
    "momentum": 0.00,
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


def _triangular_score(value: float, low: float, optimal: float, high: float) -> float:
    """Score with triangular shape peaking at optimal, zero at low and high."""
    if value <= low or value >= high:
        return 0.0
    if value <= optimal:
        if optimal == low:
            return 100.0
        return (value - low) / (optimal - low) * 100.0
    else:
        if high == optimal:
            return 100.0
        return (high - value) / (high - optimal) * 100.0


class MultiFactorScorer:
    """Score :class:`ScreeningResult` objects across value, quality, and growth dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: ScreeningResult) -> ScreeningResult:
        result.value_score = self._value_score(result)
        result.quality_score = 0.0  # placeholder; percentile will fill later
        result.growth_score = self._growth_score(result)

        # Store raw quality composite for later cross‑sectional ranking
        result._quality_raw = self._compute_quality_raw(result)

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

        # Weighted harmonic mean (will be recomputed after quality_score is set)
        result.composite_score = self._weighted_harmonic_mean(weighted_scores_to_process)
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        # 1. Compute all raw scores and guard values
        for r in results:
            self.score(r)

        # 2. Cross‑sectional percentile for the quality raw metric
        valid_results = [
            r for r in results
            if hasattr(r, '_quality_raw') and r._quality_raw is not None and r._quality_raw > 0
        ]
        if valid_results:
            valid_results.sort(key=lambda r: r._quality_raw)
            n = len(valid_results)
            for i, r in enumerate(valid_results):
                # Using (i+0.5)/n to avoid exact 0 or 100 and spread uniformly
                percentile = (i + 0.5) / n
                r.quality_score = percentile * 100.0
        # stocks without a valid raw get neutral quality score
        for r in results:
            if not hasattr(r, '_quality_raw') or r._quality_raw is None or r._quality_raw <= 0:
                r.quality_score = 50.0

        # 3. Recompute composite using final quality_score
        for r in results:
            scores = {
                "value": r.value_score,
                "quality": r.quality_score,
                "growth": r.growth_score,
                "momentum": 50.0,
            }
            weighted_scores = {
                k: v for k, v in scores.items() if self.weights.get(k, 0.0) > 0
            }
            r.composite_score = self._weighted_harmonic_mean(weighted_scores)

        # 4. Sort and assign ranks
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # New helpers for quality raw and harmonic mean
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_quality_raw(result: ScreeningResult) -> Optional[float]:
        """Compute a quality metric based on profitability and leverage; returns None if insufficient data."""
        roe = result.financials.roe
        gm = result.financials.gross_margin
        dte = result.financials.debt_to_equity
        if None in (roe, gm, dte):
            return None
        if roe <= 0 or gm <= 0:
            return None
        dte_safe = max(dte, 0.01)
        raw = (roe * gm) / (dte_safe + 1.0)
        try:
            raw = (roe * gm) / (dte_safe + 1.0) / pe_safe
        except ZeroDivisionError:
            return None
        return raw

    def _weighted_harmonic_mean(self, weighted_scores: Dict[str, float]) -> float:
        """Compute weighted harmonic mean of a dict of {factor: score}."""
        total_weight = 0.0
        sum_weight_over_score = 0.0
        MIN_SCORE = 1e-6

        for k, score_val in weighted_scores.items():
            weight = self.weights.get(k, 0.0)
            total_weight += weight
            safe_score = max(score_val, MIN_SCORE)
            sum_weight_over_score += weight / safe_score

        if sum_weight_over_score > 0:
            composite = total_weight / sum_weight_over_score
            composite = max(0.0, min(100.0, composite))
            return composite
        return 50.0

    # ------------------------------------------------------------------
    # Sub-score helpers (value and growth remain unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _value_score(result: ScreeningResult) -> float:
        weighted_scores: List[Tuple[float, float]] = []

        PB_WEIGHT = 0.20
        PS_WEIGHT = 0.14
        MARKET_CAP_WEIGHT = 0.10
        DIVIDEND_YIELD_WEIGHT = 0.12
        EV_TO_EBITDA_WEIGHT = 0.12
        PRICE_WEIGHT = 0.06
        PRICE_MARKET_CAP_WEIGHT = 0.05
        PRICE_PB_WEIGHT = 0.08
        OCF_YIELD_WEIGHT = 0.06
        CURRENT_RATIO_WEIGHT = 0.06
        PS_GM_WEIGHT = 0.06
        EARNINGS_YIELD_WEIGHT = 0.06
        EARNINGS_YIELD_ROE_WEIGHT = 0.05
        FCF_YIELD_ROE_WEIGHT = 0.05
        GP_YIELD_WEIGHT = 0.06
        DY_GM_WEIGHT = 0.04
        DTE_WEIGHT = 0.05
        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted_scores.append((_log_score(pb, best=15.0, worst=1.0), PB_WEIGHT))
        # Forward P/E ratio – lower is better
        PFORWARD_WEIGHT = 0.08
        pe_forward = result.valuation.pe_forward
        if pe_forward is not None and pe_forward > 0:
            weighted_scores.append(
                (_triangular_score(pe_forward, low=0.5, optimal=8.0, high=30.0), PFORWARD_WEIGHT)
            )
        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted_scores.append((_log_score(ps, best=0.4, worst=10.0), PS_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            score_mcap = _linear_score(market_cap, best=500_000_000_000.0, worst=1_000_000_000.0)
            weighted_scores.append((score_mcap, MARKET_CAP_WEIGHT))

        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None:
            if dividend_yield >= 0:
                weighted_scores.append(
                    (_linear_score(dividend_yield, best=0.06, worst=0.0), DIVIDEND_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, DIVIDEND_YIELD_WEIGHT))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            weighted_scores.append(
                (_log_score(ev_to_ebitda, best=4.0, worst=40.0), EV_TO_EBITDA_WEIGHT)
            )

        price = result.valuation.price
        if price is not None and price > 0:
            weighted_scores.append(
                (_log_score(price, best=2.0, worst=200.0), PRICE_WEIGHT)
            )

        if price is not None and market_cap is not None and price > 0 and market_cap > 0:
            pmc = price * market_cap
            weighted_scores.append(
                (_log_score(pmc, best=2_000_000_000.0, worst=10_000_000_000_000.0),
                 PRICE_MARKET_CAP_WEIGHT)
            )

        if price is not None and pb is not None and price > 0 and pb > 0:
            ppb = price * pb
            weighted_scores.append(
                (_log_score(ppb, best=2.0, worst=500.0), PRICE_PB_WEIGHT)
            )

        ocf = result.financials.operating_cash_flow
        if ocf is not None and market_cap is not None and market_cap > 0:
            if ocf > 0:
                ocf_yield = ocf / market_cap
                weighted_scores.append(
                    (_linear_score(ocf_yield, best=0.10, worst=0.0), OCF_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, OCF_YIELD_WEIGHT))
        # Current ratio – higher is better (liquidity)
        current_ratio = result.financials.current_ratio
        if current_ratio is not None:
            if current_ratio > 0:
                weighted_scores.append(
                    (_linear_score(current_ratio, best=3.0, worst=0.5), CURRENT_RATIO_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, CURRENT_RATIO_WEIGHT))
        gross_margin = result.financials.gross_margin
        if ps is not None and gross_margin is not None and ps > 0:
            if gross_margin > 0:
                ps_gm = ps / gross_margin
                weighted_scores.append(
                    (_log_score(ps_gm, best=0.5, worst=30.0), PS_GM_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, PS_GM_WEIGHT))

        net_income = result.financials.net_income
        if net_income is not None and market_cap is not None and market_cap > 0:
            if net_income > 0:
                ey = net_income / market_cap
                weighted_scores.append(
                    (_log_score(ey, best=0.10, worst=0.005), EARNINGS_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))
        roa_val = result.financials.roa
        if net_income is not None and market_cap is not None and roa_val is not None and market_cap > 0:
            if net_income > 0 and roa_val > 0:
                ey_roa = (net_income / market_cap) * roa_val
                weighted_scores.append(
                    (_linear_score(ey_roa, best=0.01, worst=0.0), EARNINGS_YIELD_ROE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, EARNINGS_YIELD_ROE_WEIGHT))
        else:
            weighted_scores.append((0.0, EARNINGS_YIELD_ROE_WEIGHT))

        fcf = result.financials.free_cash_flow
        if fcf is not None and market_cap is not None and roe_val is not None and market_cap > 0:
            if fcf > 0 and roe_val > 0:
                fcf_yield_roe = (fcf / market_cap) * roe_val
                weighted_scores.append(
                    (_linear_score(fcf_yield_roe, best=0.02, worst=0.0), FCF_YIELD_ROE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, FCF_YIELD_ROE_WEIGHT))
        else:
            weighted_scores.append((0.0, FCF_YIELD_ROE_WEIGHT))

        rev = result.financials.revenue
        if rev is not None and gross_margin is not None and market_cap is not None and market_cap > 0:
            if rev > 0 and gross_margin > 0:
                gp_yield = (rev * gross_margin) / market_cap
                weighted_scores.append(
                    (_log_score(gp_yield, best=0.20, worst=0.01), GP_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, GP_YIELD_WEIGHT))
        else:
            weighted_scores.append((0.0, GP_YIELD_WEIGHT))
        # New sub-factor: dividend yield × gross margin — captures profitable dividend payers
        if dividend_yield is not None and gross_margin is not None:
            if dividend_yield > 0 and gross_margin > 0:
                dy_gm = dividend_yield * gross_margin
                weighted_scores.append(
                    (_linear_score(dy_gm, best=0.02, worst=0.0), DY_GM_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, DY_GM_WEIGHT))
        else:
            weighted_scores.append((0.0, DY_GM_WEIGHT))

        # Debt-to-equity: lower leverage is better
        dte = result.financials.debt_to_equity
        if dte is not None:
            if dte > 0:
                # best (low leverage) ~0.1, worst ~5.0
                weighted_scores.append((_log_score(dte, best=0.1, worst=5.0), DTE_WEIGHT))
            else:
                weighted_scores.append((100.0, DTE_WEIGHT))
        else:
            weighted_scores.append((50.0, DTE_WEIGHT))
        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Placeholder – quality is now computed cross‑sectionally in rank()."""
        return 0.0  # will be overwritten by percentile in rank()

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        weighted_scores: List[Tuple[float, float]] = []

        PEG_GROWTH_WEIGHT = 0.5
        FORWARD_PE_IMPROVEMENT_WEIGHT = 0.5

        peg = result.valuation.peg_ratio
        if peg is not None:
            if peg > 0:
                weighted_scores.append((_log_score(peg, best=0.7, worst=1.8), PEG_GROWTH_WEIGHT))
            else:
                weighted_scores.append((0.0, PEG_GROWTH_WEIGHT))

        pe = result.valuation.pe_ratio
        pe_forward = result.valuation.pe_forward
        if pe is not None and pe_forward is not None:
            if pe_forward > 0 and pe > 0:
                forward_pe_ratio = pe / pe_forward
                weighted_scores.append(
                    (_linear_score(forward_pe_ratio, best=2.0, worst=0.5), FORWARD_PE_IMPROVEMENT_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, FORWARD_PE_IMPROVEMENT_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0