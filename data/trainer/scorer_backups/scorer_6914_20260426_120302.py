"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights – increased quality weight slightly to reward
# fundamental strength, reduced growth weight (noisy PEG/PE signals)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.65,
    "quality": 0.20,
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
        """Quality score with added current ratio factor and weight adjustments."""
        weighted_scores: List[Tuple[float, float]] = []

        ROE_WEIGHT = 0.05           # reduced from 0.10 to make room for current_ratio
        GP_A_WEIGHT = 0.25
        CROA_WEIGHT = 0.20
        EARNINGS_YIELD_WEIGHT = 0.05
        ROE_PB_WEIGHT = 0.15
        FCF_YIELD_WEIGHT = 0.15
        DEBT_TO_EQUITY_WEIGHT = 0.05
        ROA_WEIGHT = 0.05
        CURRENT_RATIO_WEIGHT = 0.05  # NEW – liquidity factor

        # 1. ROE
        roe = result.financials.roe
        if roe is not None:
            weighted_scores.append((_linear_score(roe, best=0.20, worst=0.0), ROE_WEIGHT))

        # 2. GP/A
        rev = result.financials.revenue
        gm = result.financials.gross_margin
        assets = result.financials.total_assets
        if rev is not None and gm is not None and assets is not None and assets > 0:
            gp_a = (rev * gm) / assets
            weighted_scores.append((_linear_score(gp_a, best=0.40, worst=0.0), GP_A_WEIGHT))

        # 3. CROA
        ocf = result.financials.operating_cash_flow
        if ocf is not None and assets is not None and assets > 0:
            croa = ocf / assets
            weighted_scores.append((_linear_score(croa, best=0.15, worst=0.0), CROA_WEIGHT))

        # 4. Earnings Yield (ROE / PE)
        pe = result.valuation.pe_ratio
        if roe is not None and pe is not None and pe > 0 and roe > 0:
            earnings_yield = roe / pe
            weighted_scores.append((_linear_score(earnings_yield, best=0.10, worst=0.0), EARNINGS_YIELD_WEIGHT))

        # 5. ROE / PB
        pb = result.valuation.pb_ratio
        if roe is not None and pb is not None and pb > 0 and roe > 0:
            roe_pb = roe / pb
            weighted_scores.append((_linear_score(roe_pb, best=0.10, worst=0.0), ROE_PB_WEIGHT))

        # 6. Free Cash Flow Yield (FCF / Market Cap)
        fcf = result.financials.free_cash_flow
        mcap = result.valuation.market_cap_rmb
        if fcf is not None and mcap is not None and mcap > 0 and fcf > 0:
            fcf_yield = fcf / mcap
            weighted_scores.append((_linear_score(fcf_yield, best=0.10, worst=0.0), FCF_YIELD_WEIGHT))

        # 7. Debt-to-Equity (low is good)
        dte = result.financials.debt_to_equity
        if dte is not None and dte >= 0:
            score = 100.0 - _linear_score(dte, best=0.0, worst=2.0)
            weighted_scores.append((score, DEBT_TO_EQUITY_WEIGHT))

        # 8. Return on Assets (ROA)
        roa = result.financials.roa
        if roa is not None:
            weighted_scores.append((_linear_score(roa, best=0.15, worst=0.0), ROA_WEIGHT))

        # 9. NEW: Current Ratio (liquidity, higher is better up to a point)
        cr = result.financials.current_ratio
        if cr is not None and cr >= 0:
            # Reward current ratio between 1.0 and 2.0; cap at 2.0 (no extra for >2)
            weighted_scores.append((_linear_score(cr, best=2.0, worst=1.0), CURRENT_RATIO_WEIGHT))

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