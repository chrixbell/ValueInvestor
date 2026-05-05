"""
Multi-factor scoring and ranking for screening results.

This version adds a standalone free‑cash‑flow yield component to the
value sub‑score.  Free‑cash‑flow yield (FCF / market cap) is a strong
supplementary value signal that complements the existing cash‑flow‑based
metrics.  It is scored linearly (best = 0.08, worst = 0.0) and receives
a small weight (5 %) within the value factor.  All other components,
including the geometric‑mean composite, remain unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.50,
    "quality": 0.35,
    "growth": 0.15,
    "momentum": 0.00,
}

# Per-horizon default weights — shorter horizons emphasise growth/momentum,
# longer horizons emphasise value/quality (value takes time to be recognised).
_HORIZON_WEIGHTS: Dict[str, Dict[str, float]] = {
    "1m": {"value": 0.35, "quality": 0.25, "growth": 0.30, "momentum": 0.10},
    "3m": {"value": 0.45, "quality": 0.30, "growth": 0.20, "momentum": 0.05},
    "6m": {"value": 0.50, "quality": 0.35, "growth": 0.15, "momentum": 0.00},
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

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        horizon: Optional[str] = None,
    ) -> None:
        if weights is not None:
            self.weights = weights
        elif horizon is not None and horizon in _HORIZON_WEIGHTS:
            self.weights = dict(_HORIZON_WEIGHTS[horizon])
        else:
            self.weights = dict(_DEFAULT_WEIGHTS)

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

        # Weighted geometric mean (will be recomputed after quality_score is set)
        result.composite_score = self._weighted_geometric_mean(weighted_scores_to_process)
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
            r.composite_score = self._weighted_geometric_mean(weighted_scores)

        # 4. Sort and assign ranks
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # Helpers for quality raw and geometric mean
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_raw(result: ScreeningResult) -> Optional[float]:
        """Compute a single quality‑cheapness metric; returns None if insufficient data."""
        roe = result.financials.roe
        gm = result.financials.gross_margin
        dte = result.financials.debt_to_equity
        pe = result.valuation.pe_ratio

        # Require all four inputs and strictly positive values (except DTE can be >=0)
        if None in (roe, gm, dte, pe):
            return None
        if roe <= 0 or gm <= 0 or pe <= 0:
            return None
        # avoid division by extremely small DTE (treat 0 as 0.01)
        dte_safe = max(dte, 0.01)
        pe_safe = max(pe, 1.0)
        try:
            raw = (roe * gm) / (dte_safe + 1.0) / pe_safe
        except ZeroDivisionError:
            return None
        return raw

    def _weighted_geometric_mean(self, weighted_scores: Dict[str, float]) -> float:
        """Compute weighted geometric mean of {factor: score} dict."""
        total_weight = 0.0
        sum_weight_log = 0.0
        MIN_SCORE = 1e-6

        for k, score_val in weighted_scores.items():
            weight = self.weights.get(k, 0.0)
            total_weight += weight
            safe_score = max(score_val, MIN_SCORE)
            sum_weight_log += weight * math.log(safe_score)

        if total_weight > 0:
            composite = math.exp(sum_weight_log / total_weight)
            return max(0.0, min(100.0, composite))
        return 50.0

    # ------------------------------------------------------------------
    # Sub-score helpers
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
        PS_GM_WEIGHT = 0.06
        EARNINGS_YIELD_WEIGHT = 0.06
        EARNINGS_YIELD_ROE_WEIGHT = 0.05
        FCF_YIELD_ROE_WEIGHT = 0.05
        GP_YIELD_WEIGHT = 0.06
        GP_TA_WEIGHT = 0.04  # gross profit / total assets
        ROA_WEIGHT = 0.04    # return on assets
        FCF_YIELD_WEIGHT = 0.05  # **new** standalone free-cash-flow yield

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted_scores.append((_log_score(pb, best=15.0, worst=1.0), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted_scores.append((_log_score(ps, best=0.4, worst=10.0), PS_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            # Inverted: smaller market cap = higher score (small‑cap premium)
            score_mcap = _linear_score(market_cap, best=1_000_000_000.0, worst=500_000_000_000.0)
            weighted_scores.append((score_mcap, MARKET_CAP_WEIGHT))

        # Dividend yield with free‑cash‑flow‑coverage adjustment
        dividend_yield = result.valuation.dividend_yield
        fcf = result.financials.free_cash_flow

        if dividend_yield is not None:
            base_dy_score = _linear_score(dividend_yield, best=0.06, worst=0.0) if dividend_yield >= 0 else 0.0
            # Adjust for free‑cash‑flow coverage if both are positive
            if dividend_yield > 0 and fcf is not None and market_cap is not None and market_cap > 0 and fcf > 0:
                fcf_yield = fcf / market_cap
                coverage = fcf_yield / dividend_yield
                if coverage >= 1.0:
                    cohesion_factor = 1.0
                else:
                    # Linear penalty for insufficient coverage
                    cohesion_factor = coverage
                weighted_scores.append((base_dy_score * cohesion_factor, DIVIDEND_YIELD_WEIGHT))
            else:
                # If no positive free cash flow, dividend is penalised
                weighted_scores.append((0.0, DIVIDEND_YIELD_WEIGHT))
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

        roe_val = result.financials.roe
        if net_income is not None and market_cap is not None and roe_val is not None and market_cap > 0:
            if net_income > 0 and roe_val > 0:
                ey_roe = (net_income / market_cap) * roe_val
                weighted_scores.append(
                    (_linear_score(ey_roe, best=0.02, worst=0.0), EARNINGS_YIELD_ROE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, EARNINGS_YIELD_ROE_WEIGHT))
        else:
            weighted_scores.append((0.0, EARNINGS_YIELD_ROE_WEIGHT))

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

        # Gross profit / total assets (asset profitability)
        total_assets = result.financials.total_assets
        if rev is not None and gross_margin is not None and total_assets is not None and rev > 0 and gross_margin > 0 and total_assets > 0:
            gp_ta = (rev * gross_margin) / total_assets
            weighted_scores.append((_linear_score(gp_ta, best=0.30, worst=0.0), GP_TA_WEIGHT))
        else:
            weighted_scores.append((0.0, GP_TA_WEIGHT))

        # Return on assets (ROA)
        roa = result.financials.roa
        if roa is not None and roa > 0:
            weighted_scores.append((_linear_score(roa, best=0.15, worst=0.0), ROA_WEIGHT))
        else:
            weighted_scores.append((0.0, ROA_WEIGHT))

        # **NEW** Standalone free‑cash‑flow yield (FCF / market cap)
        if fcf is not None and market_cap is not None and market_cap > 0:
            if fcf > 0:
                fcf_yield_standalone = fcf / market_cap
                weighted_scores.append(
                    (_linear_score(fcf_yield_standalone, best=0.08, worst=0.0),
                     FCF_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, FCF_YIELD_WEIGHT))
        else:
            weighted_scores.append((0.0, FCF_YIELD_WEIGHT))

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