"""Multi-factor scoring and ranking for screening results."""

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

        # Weighted harmonic mean
        total_weight = 0.0
        sum_weight_over_score = 0.0
        MIN_SCORE = 1e-6

        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            safe_score = max(score_val, MIN_SCORE)
            sum_weight_over_score += weight / safe_score

        if sum_weight_over_score > 0:
            composite_score = total_weight / sum_weight_over_score
        else:
            composite_score = 50.0

        composite_score = max(0.0, min(100.0, composite_score))
        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        for r in results:
            self.score(r)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _value_score(result: ScreeningResult) -> float:
        weighted_scores: List[Tuple[float, float]] = []

        PB_WEIGHT = 0.20
        PS_WEIGHT = 0.14
        MARKET_CAP_WEIGHT = 0.01
        DIVIDEND_YIELD_WEIGHT = 0.12
        EV_TO_EBITDA_WEIGHT = 0.12
        PRICE_WEIGHT = 0.06
        PRICE_MARKET_CAP_WEIGHT = 0.05
        PRICE_PB_WEIGHT = 0.08
        OCF_YIELD_WEIGHT = 0.06
        PS_GM_WEIGHT = 0.06
        EARNINGS_YIELD_WEIGHT = 0.06

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted_scores.append((_log_score(pb, best=15.0, worst=1.0), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted_scores.append((_log_score(ps, best=0.4, worst=10.0), PS_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            # Triangular score: reward mid‑cap stocks, penalise very small and very large
            weighted_scores.append(
                (
                    _triangular_score(
                        market_cap,
                        low=500_000_000.0,      # 0.5 B RMB – tiny/micro
                        optimal=30_000_000_000.0, # 30 B RMB – sweet spot
                        high=150_000_000_000.0,   # 150 B RMB – large/mega
                    ),
                    MARKET_CAP_WEIGHT,
                )
            )

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

        gross_margin = result.financials.gross_margin
        if ps is not None and gross_margin is not None and ps > 0:
            if gross_margin > 0:
                ps_gm = ps / gross_margin
                weighted_scores.append(
                    (_log_score(ps_gm, best=0.5, worst=30.0), PS_GM_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, PS_GM_WEIGHT))

        # Earnings yield (E/P) – net income over market cap
        net_income = result.financials.net_income
        if net_income is not None and market_cap is not None and market_cap > 0:
            if net_income > 0:
                ey = net_income / market_cap
                weighted_scores.append(
                    (_log_score(ey, best=0.10, worst=0.005), EARNINGS_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Quality score captures profitability, efficiency, safety, and ROE/PE interaction."""
        weighted_scores: List[Tuple[float, float]] = []

        ROE_WEIGHT = 0.15
        GP_A_WEIGHT = 0.10
        ROA_WEIGHT = 0.15
        FCF_YIELD_WEIGHT = 0.10
        CURRENT_RATIO_WEIGHT = 0.10
        DEBT_TO_EQUITY_WEIGHT = 0.10
        FCF_REVENUE_WEIGHT = 0.10
        PE_DTE_WEIGHT = 0.10
        NET_MARGIN_WEIGHT = 0.10
        ROE_PE_WEIGHT = 0.10

        roe = result.financials.roe
        if roe is not None:
            if roe >= 0:
                # Triangular hump: peaks at 15 % ROE, falls to zero at 0 % and 50 %
                weighted_scores.append(
                    (_triangular_score(roe, low=0.0, optimal=0.15, high=0.50), ROE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, ROE_WEIGHT))

        rev = result.financials.revenue
        gm = result.financials.gross_margin
        assets = result.financials.total_assets
        if rev is not None and gm is not None and assets is not None and assets > 0:
            gp_a = (rev * gm) / assets
            if gp_a >= 0:
                weighted_scores.append((_linear_score(gp_a, best=0.40, worst=0.0), GP_A_WEIGHT))
            else:
                weighted_scores.append((0.0, GP_A_WEIGHT))

        fcf = result.financials.free_cash_flow
        mcap = result.valuation.market_cap_rmb
        if fcf is not None and mcap is not None and mcap > 0:
            if fcf > 0:
                fcf_yield = fcf / mcap
                weighted_scores.append((_linear_score(fcf_yield, best=0.10, worst=0.0), FCF_YIELD_WEIGHT))
            else:
                weighted_scores.append((0.0, FCF_YIELD_WEIGHT))

        roa = result.financials.roa
        if roa is not None:
            if roa >= 0:
                weighted_scores.append((_linear_score(roa, best=0.15, worst=0.0), ROA_WEIGHT))
            else:
                weighted_scores.append((0.0, ROA_WEIGHT))

        current_ratio = result.financials.current_ratio
        if current_ratio is not None:
            if current_ratio > 0:
                weighted_scores.append(
                    (_linear_score(current_ratio, best=3.0, worst=0.5), CURRENT_RATIO_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, CURRENT_RATIO_WEIGHT))

        # Debt-to-equity: replaced symmetric hump with exponential penalty.
        # Low leverage is preferred; score decays as D/E rises.
        dte = result.financials.debt_to_equity
        if dte is not None:
            if dte >= 0:
                score = 100.0 * math.exp(-dte / 0.8)
                weighted_scores.append((score, DEBT_TO_EQUITY_WEIGHT))
            else:
                weighted_scores.append((0.0, DEBT_TO_EQUITY_WEIGHT))

        if fcf is not None and rev is not None and rev > 0:
            if fcf > 0:
                fcf_rev = fcf / rev
                weighted_scores.append(
                    (_linear_score(fcf_rev, best=0.30, worst=0.0), FCF_REVENUE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, FCF_REVENUE_WEIGHT))

        pe = result.valuation.pe_ratio
        if pe is not None and dte is not None:
            if pe >= 0 and dte >= 0:
                pe_dte = pe * dte
                # Triangular hump: moderate PE × DTE is best (e.g. 8),
                # very low or very high product is penalised.
                weighted_scores.append(
                    (_triangular_score(pe_dte, low=0.0, optimal=8.0, high=50.0), PE_DTE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, PE_DTE_WEIGHT))

        net_margin = result.financials.net_margin
        if net_margin is not None:
            if net_margin >= 0:
                weighted_scores.append(
                    (_linear_score(net_margin, best=0.30, worst=0.0), NET_MARGIN_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, NET_MARGIN_WEIGHT))

        if roe is not None and pe is not None and pe > 0:
            if roe > 0:
                roe_pe = roe / pe
                weighted_scores.append(
                    (_linear_score(roe_pe, best=0.05, worst=0.0), ROE_PE_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, ROE_PE_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

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