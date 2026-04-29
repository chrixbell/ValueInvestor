"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Weights remain unchanged – the single modification is inside the quality score.
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

        DIVIDEND_YIELD_WEIGHT = 0.2
        PB_WEIGHT = 0.3
        PS_WEIGHT = 0.2
        MARKET_CAP_WEIGHT = 0.1
        EV_TO_EBITDA_WEIGHT = 0.2

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted_scores.append((_log_score(pb, best=15.0, worst=1.0), PB_WEIGHT))

        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted_scores.append((_log_score(ps, best=0.4, worst=10.0), PS_WEIGHT))

        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            weighted_scores.append(
                (
                    _log_score(
                        market_cap,
                        best=1_000_000_000.0,
                        worst=60_000_000_000.0,
                    ),
                    MARKET_CAP_WEIGHT,
                )
            )

        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield >= 0:
            weighted_scores.append(
                (_linear_score(dividend_yield, best=0.06, worst=0.0), DIVIDEND_YIELD_WEIGHT)
            )

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            weighted_scores.append(
                (_log_score(ev_to_ebitda, best=4.0, worst=40.0), EV_TO_EBITDA_WEIGHT)
            )

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _quality_score(result: ScreeningResult) -> float:
        """Quality score concentrated on ROE, ROA, ROE/PB, FCF yield,
        current ratio, and low leverage."""
        weighted_scores: List[Tuple[float, float]] = []

        ROE_WEIGHT = 0.15          # reduced from 0.20 to make room for current ratio
        GP_A_WEIGHT = 0.10
        ROA_WEIGHT = 0.15
        ROE_PB_WEIGHT = 0.20
        FCF_YIELD_WEIGHT = 0.20
        CURRENT_RATIO_WEIGHT = 0.10   # new factor
        DEBT_TO_EQUITY_WEIGHT = 0.10

        # 1. ROE
        roe = result.financials.roe
        if roe is not None:
            weighted_scores.append((_linear_score(roe, best=0.20, worst=0.0), ROE_WEIGHT))

        # 2. GP/A (gross profitability / assets)
        rev = result.financials.revenue
        gm = result.financials.gross_margin
        assets = result.financials.total_assets
        if rev is not None and gm is not None and assets is not None and assets > 0:
            gp_a = (rev * gm) / assets
            weighted_scores.append((_linear_score(gp_a, best=0.40, worst=0.0), GP_A_WEIGHT))

        # 3. ROE / PB (quality adjusted for valuation)
        pb = result.valuation.pb_ratio
        if roe is not None and pb is not None and pb > 0 and roe > 0:
            roe_pb = roe / pb
            weighted_scores.append((_linear_score(roe_pb, best=0.10, worst=0.0), ROE_PB_WEIGHT))

        # 4. Free Cash Flow Yield
        fcf = result.financials.free_cash_flow
        mcap = result.valuation.market_cap_rmb
        if fcf is not None and mcap is not None and mcap > 0 and fcf > 0:
            fcf_yield = fcf / mcap
            weighted_scores.append((_linear_score(fcf_yield, best=0.10, worst=0.0), FCF_YIELD_WEIGHT))

        # 5. Return on Assets (ROA)
        roa = result.financials.roa
        if roa is not None:
            weighted_scores.append((_linear_score(roa, best=0.15, worst=0.0), ROA_WEIGHT))

        # 6. Current ratio (liquidity)
        current_ratio = result.financials.current_ratio
        if current_ratio is not None and current_ratio > 0:
            # Higher is better; typical healthy range 1.5-3.0
            weighted_scores.append(
                (_linear_score(current_ratio, best=3.0, worst=0.5), CURRENT_RATIO_WEIGHT)
            )

        # 7. Debt-to-Equity (low is good)
        dte = result.financials.debt_to_equity
        if dte is not None and dte >= 0:
            score = 100.0 - _linear_score(dte, best=0.0, worst=2.0)
            weighted_scores.append((score, DEBT_TO_EQUITY_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        weighted_scores: List[Tuple[float, float]] = []

        PEG_GROWTH_WEIGHT = 0.4
        PE_GROWTH_WEIGHT = 0.3
        FORWARD_PE_IMPROVEMENT_WEIGHT = 0.3

        peg = result.valuation.peg_ratio
        if peg is not None and peg > 0:
            weighted_scores.append((_log_score(peg, best=0.7, worst=1.8), PEG_GROWTH_WEIGHT))

        pe = result.valuation.pe_ratio
        if pe is not None and pe > 0:
            weighted_scores.append((_log_score(pe, best=150.0, worst=10.0), PE_GROWTH_WEIGHT))

        pe_forward = result.valuation.pe_forward
        if pe is not None and pe_forward is not None and pe_forward > 0 and pe > 0:
            forward_pe_ratio = pe / pe_forward
            weighted_scores.append(
                (_linear_score(forward_pe_ratio, best=2.0, worst=0.5), FORWARD_PE_IMPROVEMENT_WEIGHT)
            )

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0