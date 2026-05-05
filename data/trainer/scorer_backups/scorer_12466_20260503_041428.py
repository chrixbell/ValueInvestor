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
        result.growth_score = self._growth_score(result)
        # Compute and store raw growth metric; will be percentile‑ranked later
        result._growth_raw = self._compute_growth_raw(result)

        result._quality_raw = self._compute_quality_raw(result)
        # Direct quality score for individual use; rank() overwrites with cross‑sectional percentile
        if result._quality_raw is not None and result._quality_raw > 0:
            result.quality_score = _log_score(result._quality_raw, best=0.05, worst=0.001)
        else:
            result.quality_score = 50.0

        result._momentum_score = self._momentum_score(result)
        scores = {
            "value": result.value_score,
            "quality": result.quality_score,
            "growth": result.growth_score,
            "momentum": result._momentum_score,
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
        # Apply loss penalty
        result.composite_score *= self._loss_penalty(result)
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
                linear = percentile * 100.0
                # Compress quality score range to reduce dominance; helps multi-horizon robustness
                r.quality_score = 50.0 + (linear - 50.0) * 0.85
        # stocks without a valid raw get neutral quality score
        for r in results:
            if not hasattr(r, '_quality_raw') or r._quality_raw is None or r._quality_raw <= 0:
                r.quality_score = 50.0
        # 2b. Cross‑sectional percentile for growth raw metric
        valid_growth = [
            r for r in results
            if hasattr(r, '_growth_raw') and r._growth_raw is not None and r._growth_raw > 0
        ]
        if valid_growth:
            valid_growth.sort(key=lambda r: r._growth_raw)
            n = len(valid_growth)
            for i, r in enumerate(valid_growth):
                # Spread uniformly (avoid 0 or 100)
                percentile = (i + 0.5) / n
                r.growth_score = percentile * 100.0
        for r in results:
            if not hasattr(r, '_growth_raw') or r._growth_raw is None or r._growth_raw <= 0:
                r.growth_score = 50.0

        # 3. Recompute composite using final quality_score
        for r in results:
            synergy = min(r.value_score, r.quality_score)
            value_growth = math.sqrt(max(r.value_score, 0.0) * max(r.growth_score, 0.0))
            scores = {
                "value": r.value_score,
                "quality": r.quality_score,
                "growth": r.growth_score,
                "momentum": getattr(r, "_momentum_score", 50.0),
                "synergy": synergy,
                "value_growth": value_growth,
            }
            weighted_scores = {
                k: v for k, v in scores.items() if self.weights.get(k, 0.0) > 0
            }
            r.composite_score = self._weighted_geometric_mean(weighted_scores)
            # Apply loss penalty
            r.composite_score *= self._loss_penalty(r)

        # 4. Sort and assign ranks
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx
        return results
    # ------------------------------------------------------------------
    # New momentum sub‑score – financial strength (free cash flow yield × current ratio)
    # ------------------------------------------------------------------
    @staticmethod
    def _momentum_score(result: ScreeningResult) -> float:
        """Asset turnover (revenue / total assets) – higher is better.
        Measures efficiency of asset usage.
        """
        revenue = result.financials.revenue
        total_assets = result.financials.total_assets
        if revenue is not None and total_assets is not None and total_assets > 0:
            if revenue > 0:
                turnover = revenue / total_assets
                return _linear_score(turnover, best=2.0, worst=0.0)
        return 50.0

    # ------------------------------------------------------------------
    # New helpers for quality raw and geometric mean
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_quality_raw(result: ScreeningResult) -> Optional[float]:
        roe = result.financials.roe
        gm = result.financials.gross_margin
        dte = result.financials.debt_to_equity
        if None in (roe, gm, dte):
            return None
        if roe <= 0 or gm <= 0.15:
            return None
        roe = min(roe, 0.30)
        dte_pos = max(dte, 0.0)
        if dte_pos > 2.0:
            return None
        raw = (roe * gm) / (1.0 + math.sqrt(dte_pos))
        # Reward quality with strong operating cash flow relative to assets
        ocf = result.financials.operating_cash_flow
        total_assets = result.financials.total_assets
        if (ocf is not None and total_assets is not None
                and total_assets > 0 and ocf > 0):
            ocfy = ocf / total_assets
            # Cap ocfy to avoid extreme multiples
            ocfy = min(ocfy, 0.5)
            raw *= (1.0 + ocfy)
        return raw
    @staticmethod
    def _compute_growth_raw(result: ScreeningResult) -> Optional[float]:
        """Compute a growth raw metric from PEG ratio.

        Lower PEG is better, so we invert it (1/PEG) so that higher raw means
        better growth at a reasonable price.
        """
        peg = result.valuation.peg_ratio
        if peg is None or peg <= 0:
            return None
        return 1.0 / peg

    # ------------------------------------------------------------------
    def _weighted_geometric_mean(self, weighted_scores: Dict[str, float]) -> float:
        """Compute weighted geometric mean of a dict of {factor: score}."""
        total_weight = 0.0
        log_sum = 0.0
        MIN_SCORE = 10.0

        for k, score_val in weighted_scores.items():
            weight = self.weights.get(k, 0.0)
            total_weight += weight
            safe_score = max(score_val, MIN_SCORE)
            log_sum += weight * math.log(safe_score)

        if total_weight > 0:
            composite = math.exp(log_sum / total_weight)
            composite = max(0.0, min(100.0, composite))
            return composite
        return 50.0

    # ------------------------------------------------------------------
    # Loss penalty
    # ------------------------------------------------------------------
    @staticmethod
    def _loss_penalty(result: ScreeningResult) -> float:
        """Penalise loss-making firms by scaling composite score down."""
        net_income = result.financials.net_income
        market_cap = result.valuation.market_cap_rmb
        if net_income is not None and market_cap is not None and market_cap > 0 and net_income < 0:
            return 1.0 / (1.0 + 5.0 * abs(net_income) / market_cap)
        return 1.0

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
        EARNINGS_YIELD_WEIGHT = 0.06
        pb = result.valuation.pb_ratio
        if pb is not None and pb > 0:
            weighted_scores.append((_log_score(pb, best=20.0, worst=0.5), PB_WEIGHT))
        # Forward P/E ratio – lower is better
        PFORWARD_WEIGHT = 0.08
        pe_forward = result.valuation.pe_forward
        if pe_forward is not None and pe_forward > 0:
            weighted_scores.append(
                (_log_score(pe_forward, best=10.0, worst=50.0), PFORWARD_WEIGHT)
            )
        # Trailing-to-forward PE improvement: higher ratio indicates earnings growth or undervaluation
        FORWARD_PE_IMPROVEMENT_VAL_WEIGHT = 0.06
        pe = result.valuation.pe_ratio
        if pe is not None and pe_forward is not None and pe > 0 and pe_forward > 0:
            fw_pe_improve = pe / pe_forward
            weighted_scores.append(
                (_linear_score(fw_pe_improve, best=1.5, worst=0.5), FORWARD_PE_IMPROVEMENT_VAL_WEIGHT)
            )
        ps = result.valuation.ps_ratio
        if ps is not None and ps > 0:
            weighted_scores.append((_log_score(ps, best=0.15, worst=4.0), PS_WEIGHT))
        market_cap = result.valuation.market_cap_rmb
        if market_cap is not None and market_cap > 0:
            score_mcap = _linear_score(market_cap, best=500_000_000_000.0, worst=1_000_000_000.0)
            weighted_scores.append((score_mcap, MARKET_CAP_WEIGHT))
        dividend_yield = result.valuation.dividend_yield
        if dividend_yield is not None:
            if dividend_yield >= 0:
                weighted_scores.append(
                    (_triangular_score(dividend_yield, low=0.0, optimal=0.03, high=0.08), DIVIDEND_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, DIVIDEND_YIELD_WEIGHT))

        ev_to_ebitda = result.valuation.ev_to_ebitda
        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            weighted_scores.append(
                (_triangular_score(ev_to_ebitda, low=2.0, optimal=8.0, high=22.0), EV_TO_EBITDA_WEIGHT)
            )
        # ROE / EV/EBITDA: reward companies that are profitable relative to enterprise valuation
        roe_val_val = result.financials.roe
        if ev_to_ebitda is not None and roe_val_val is not None and ev_to_ebitda > 0 and roe_val_val > 0:
            roe_ey = roe_val_val / ev_to_ebitda
            weighted_scores.append(
                (_linear_score(roe_ey, best=0.5, worst=0.01), 0.06)
            )
        price = result.valuation.price
        if price is not None and price > 0:
            weighted_scores.append(
                (_log_score(price, best=8.0, worst=60.0), PRICE_WEIGHT)
            )
        if price is not None and market_cap is not None and price > 0 and market_cap > 0:
            pmc = price * market_cap
            weighted_scores.append(
                (_log_score(pmc, best=10_000_000_000.0, worst=1_000_000_000_000.0),
                 PRICE_MARKET_CAP_WEIGHT)
            )

        if price is not None and pb is not None and price > 0 and pb > 0:
            ppb = price * pb
            weighted_scores.append(
                (_log_score(ppb, best=5.0, worst=250.0), PRICE_PB_WEIGHT)
            )

        ocf = result.financials.operating_cash_flow
        if ocf is not None and market_cap is not None and market_cap > 0:
            if ocf > 0:
                ocf_yield = ocf / market_cap
                weighted_scores.append(
                    (_log_score(ocf_yield, best=0.03, worst=0.0005), OCF_YIELD_WEIGHT)
                )
            else:
                weighted_scores.append((0.0, OCF_YIELD_WEIGHT))
        gross_margin = result.financials.gross_margin
        net_income = result.financials.net_income
        if net_income is not None and market_cap is not None and market_cap > 0:
            if net_income > 0:
                ey = net_income / market_cap
                weighted_scores.append(
                    (_log_score(ey, best=0.02, worst=0.0005), EARNINGS_YIELD_WEIGHT)
                )
                weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))
            else:
                weighted_scores.append((0.0, EARNINGS_YIELD_WEIGHT))
        if weighted_scores:
            total_score = sum(score * weight for score, weight in weighted_scores)
            total_weight = sum(weight for score, weight in weighted_scores)
            raw_avg = total_score / total_weight if total_weight > 0 else 50.0
            if raw_avg <= 0:
                return 0.0
            boost_alpha = 1.55
            boosted = (raw_avg ** boost_alpha) / (100.0 ** (boost_alpha - 1.0))
            return max(0.0, min(100.0, boosted))
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
