"""Multi-factor scoring and ranking for screening results."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from valueinvestor.data.models import ScreeningResult

logger = logging.getLogger(__name__)

# Default factor weights — value-heavy but with a small quality tilt
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "value": 0.55,
    "quality": 0.30,
    "growth": 0.15,
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

        # Collect scores with positive weights.
        weighted_scores_to_process = {}
        for k, score_val in scores.items():
            weight = self.weights.get(k, 0.0)
            if weight > 0:
                weighted_scores_to_process[k] = score_val

        if not weighted_scores_to_process:
            result.composite_score = 50.0
            return result

        # Weighted harmonic mean of the sub-scores.
        total_weight = 0.0
        sum_weighted_inverse = 0.0
        for k, score_val in weighted_scores_to_process.items():
            weight = self.weights[k]
            total_weight += weight
            safe_score = max(score_val, 1.0)
            sum_weighted_inverse += weight / safe_score

        if total_weight > 0:
            composite_score = total_weight / sum_weighted_inverse
        else:
            composite_score = 50.0

        composite_score = max(0.0, min(100.0, composite_score))
        result.composite_score = composite_score
        return result

    def rank(self, results: List[ScreeningResult]) -> List[ScreeningResult]:
        """Score every result, sort by composite descending, and assign ranks."""
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
        """Combination of traditional and cash-flow-based value metrics."""
        weighted_scores: List[Tuple[float, float]] = []

        # Sub-factor weights (total = 1.0)
        PB_WEIGHT = 0.15
        EARNINGS_YIELD_WEIGHT = 0.10          # ROE / PB
        DIRECT_EY_WEIGHT = 0.15               # Net Income / Market Cap
        MARKET_CAP_WEIGHT = 0.10
        DIVIDEND_YIELD_WEIGHT = 0.15
        EV_TO_EBITDA_WEIGHT = 0.15
        FCF_YIELD_WEIGHT = 0.20               # Increased to give more emphasis on cash generation

        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            # Higher PB has empirically been better in this ranking context
            weighted_scores.append((_log_score(pb, best=18.0, worst=1.5), PB_WEIGHT))

        # Earnings Yield (ROE / PB)
        roe = result.financials.roe
        if roe is not None and pb is not None and pb > 0:
            earnings_yield = roe / pb
            weighted_scores.append((_linear_score(earnings_yield, best=0.15, worst=0.02), EARNINGS_YIELD_WEIGHT))

        # Direct Earnings Yield (Net Income / Market Cap)
        net_income = result.financials.net_income
        market_cap = result.valuation.market_cap_rmb
        if net_income is not None and market_cap is not None and market_cap > 0:
            if net_income > 0:
                ey_direct = net_income / market_cap
                weighted_scores.append((_linear_score(ey_direct, best=0.10, worst=0.0), DIRECT_EY_WEIGHT))
            else:
                # Negative earnings → poor value signal
                weighted_scores.append((0.0, DIRECT_EY_WEIGHT))

        # Market cap (small-cap tilt)
        if market_cap is not None and market_cap > 0:
            weighted_scores.append((_log_score(market_cap, best=2_000_000_000.0, worst=60_000_000_000.0), MARKET_CAP_WEIGHT))

        # Dividend Yield
        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None and dividend_yield >= 0:
            weighted_scores.append((_linear_score(dividend_yield, best=0.05, worst=0.01), DIVIDEND_YIELD_WEIGHT))

        # EV/EBITDA
        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            weighted_scores.append((_linear_score(ev_to_ebitda, best=5.0, worst=15.0), EV_TO_EBITDA_WEIGHT))

        # Free Cash Flow Yield (FCF / Market Cap) – handle negative FCF by assigning score 0
        fcf = result.financials.free_cash_flow
        if fcf is not None and market_cap is not None and market_cap > 0:
            if fcf >= 0:
                fcf_yield = fcf / market_cap
                weighted_scores.append((_linear_score(fcf_yield, best=0.10, worst=0.0), FCF_YIELD_WEIGHT))
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
        """Quality score focusing on ROE, efficiency, financial health, and liquidity."""
        weighted_scores: List[Tuple[float, float]] = []

        # Quality sub-factors weights (Total = 1.0)
        ROE_WEIGHT = 0.20
        DEBT_TO_EQUITY_WEIGHT = 0.15
        CURRENT_RATIO_WEIGHT = 0.10
        OCF_MARGIN_WEIGHT = 0.15
        GROSS_PROFITABILITY_WEIGHT = 0.15
        OCF_TO_NET_INCOME_WEIGHT = 0.10
        ASSET_TURNOVER_WEIGHT = 0.15

        # Extract core fields
        roe = result.financials.roe
        debt_ratio = result.financials.debt_to_equity
        current_ratio = result.financials.current_ratio
        operating_cash_flow = result.financials.operating_cash_flow
        revenue = result.financials.revenue
        gross_margin = result.financials.gross_margin
        total_assets = result.financials.total_assets
        net_income = result.financials.net_income

        # 1. Profitability: ROE
        if roe is not None:
            weighted_scores.append((_linear_score(roe, best=0.20, worst=0.0), ROE_WEIGHT))

        # 2. Leverage: Debt-to-Equity
        if debt_ratio is not None:
            if debt_ratio == 0:
                score = 100.0
            elif debt_ratio > 0:
                score = _log_score(debt_ratio, best=0.5, worst=2.0)
            else:
                score = 0.0
            weighted_scores.append((score, DEBT_TO_EQUITY_WEIGHT))

        # 3. Liquidity: Current Ratio
        if current_ratio is not None and current_ratio > 0:
            if current_ratio < 1.0:
                score = _linear_score(current_ratio, best=1.0, worst=0.5)
            elif current_ratio > 4.0:
                score = _linear_score(current_ratio, best=4.0, worst=6.0)
                score = 100.0 - score
            else:
                score = _linear_score(current_ratio, best=2.0, worst=1.0) if current_ratio <= 2.0 else _linear_score(current_ratio, best=2.0, worst=4.0)
            weighted_scores.append((score, CURRENT_RATIO_WEIGHT))

        # 4. Cash Flow: Operating Cash Flow Margin
        if operating_cash_flow is not None and revenue is not None and revenue > 0:
            ocf_margin = operating_cash_flow / revenue
            weighted_scores.append((_linear_score(ocf_margin, best=0.20, worst=0.0), OCF_MARGIN_WEIGHT))

        # 5. Efficiency: Gross Profitability (Novy-Marx)
        if (revenue is not None and gross_margin is not None and 
            total_assets is not None and total_assets > 0):
            gp_assets = (revenue * gross_margin) / total_assets
            weighted_scores.append((_linear_score(gp_assets, best=0.30, worst=0.05), GROSS_PROFITABILITY_WEIGHT))

        # 6. Earnings Quality: OCF / Net Income
        if operating_cash_flow is not None and net_income is not None and net_income > 0:
            ocf_to_ni = operating_cash_flow / net_income
            weighted_scores.append((_linear_score(ocf_to_ni, best=1.2, worst=0.4), OCF_TO_NET_INCOME_WEIGHT))

        # 7. Efficiency: Asset Turnover
        if revenue is not None and total_assets is not None and total_assets > 0:
            asset_turnover = revenue / total_assets
            weighted_scores.append((_linear_score(asset_turnover, best=1.5, worst=0.2), ASSET_TURNOVER_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0

    @staticmethod
    def _growth_score(result: ScreeningResult) -> float:
        """Growth score based on PEG, PE (proxy for expectations), and forward improvement."""
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
            weighted_scores.append((_linear_score(forward_pe_ratio, best=2.0, worst=0.5), FORWARD_PE_IMPROVEMENT_WEIGHT))

        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            return total_score / total_weight if total_weight > 0 else 50.0
        else:
            return 50.0